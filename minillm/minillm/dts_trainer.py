import json
import os
import deepspeed
from time import time
from typing import Optional, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    mpu)

from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from .utils import (
    get_log_probs,
    get_rev_kl,
    significant
)

from .model import (
    PPOModel
)

from .pipelines import PPOPipeline, LMPipeline

from .storages import PPORolloutStorage
from .losses import Loss

from utils import print_rank, save_rank, get_rank, all_gather, save_parallel
from rouge_metric import compute_metrics

from .trainer import PPOTrainer

class DynamicTemperatureScheduler(PPOTrainer):
    def __init__(
        self,
        args,
        tokenizer,
        reward_fn,
        ds_config
    ):
        super(DynamicTemperatureScheduler, self).__init__(args, tokenizer, reward_fn, ds_config)

        self.initial_temperature = args.initial_temperature
        self.current_temperature = self.initial_temperature
        self.min_temperature = args.min_temperature
        self.max_temperature = args.max_temperature
        self.max_epochs = args.training_epochs
        self.has_temp = True
        self.adjust_temp = args.adjust_temp
        self.curve_shape = args.curve_shape
        args.temperature = self.current_temperature

    def update_temperature(self, current_epoch, loss_divergence):
        progress = torch.tensor(current_epoch / self.max_epochs)
        cosine_factor = 0.5 * (1 + torch.cos(self.curve_shape * torch.pi * progress))

        if self.adjust_temp is True:
            adaptive_scale = loss_divergence / (loss_divergence + 1)

            if adaptive_scale > 1:
                if adaptive_scale > 2:
                    adaptive_scale = 1.35
                target_temperature = self.initial_temperature * cosine_factor * (adaptive_scale)
            else:
                target_temperature = self.initial_temperature * cosine_factor
        else:
            target_temperature = self.initial_temperature * cosine_factor

        target_temperature = torch.clamp(
            target_temperature,
            self.min_temperature,
            self.max_temperature
        )

        # target_temperature = round(target_temperature.item(), 2)

        momentum = 0.9
        self.current_temperature = momentum * self.current_temperature + (1 - momentum) * target_temperature
        self.args.temperature = self.current_temperature

    def get_temperature(self):
        """
        Retrieve current temperature value.

        Returns:
            float: Current dynamic temperature.
        """

        return self.current_temperature

    def compute_logits_and_log_probs(
            self, query_ids: torch.Tensor, response_ids: torch.Tensor,
            inf_mask: Optional[torch.Tensor] = None, base: bool = "base", return_logprobs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:

        batch = self.get_model_inputs(
            query_ids, response_ids
        )

        if base == "base":
            model_cls = self.model.module.forward
        elif base == "teacher":
            model_cls = self.teacher_model
        else:
            raise NotImplementedError

        outputs = model_cls(
            **batch,
            return_dict=True,
            use_cache=False
        )

        logits = outputs.logits
        logits = logits / self.current_temperature

        start = query_ids.size(1) - 1
        end = query_ids.size(1) + response_ids.size(1) - 1
        logits = logits[:, start:end]

        if inf_mask is not None:
            logits = logits.masked_fill(inf_mask, -float("inf"))

        mask = batch["attention_mask"][:, start:end]

        if return_logprobs:
            logprobs = get_log_probs(logits, response_ids, mask, inf_mask, model_parallel=self.args.model_parallel)
            return logits, logprobs

        return logits

    def train(self):
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """

        self.prepare_learning()
        self.iter_count = 1
        self.global_iter_count = 1
        self.nth_evaluation = 0

        self.evaluate()

        print_rank("Total Steps:", self.total_steps, "Data Epochs:", self.args.epochs)
        lm_epochs = 0
        logging_stats = defaultdict(float)

        for training_epoch in range(self.args.training_epochs):
            for ppo_epoch in range(self.n_updates_per_batch):
                for it, batch in enumerate(self.train_dataloader):
                    if self.lm_pipeline is not None:
                        try:
                            lm_batch = next(self.lm_iterator)
                        except StopIteration:
                            lm_epochs += 1
                            print_rank(f"Another lm epoch, lm epochs: {lm_epochs}")
                            save_rank(f"Another lm epoch, lm epochs: {lm_epochs}",
                                      os.path.join(self.args.save, "log.txt"))
                            self.lm_dataloader.sampler.set_epoch(lm_epochs)
                            self.lm_iterator = iter(self.lm_dataloader)
                            lm_batch = next(self.lm_iterator)

                        self.lm_pipeline.move_to_device(*lm_batch, self.device)
                    else:
                        lm_batch = None

                    self.store.move_to_device(batch, self.device)
                    stats = {}

                    if self.args.model_parallel:
                        self.store.broadcast(batch, src=mpu.get_model_parallel_src_rank(),
                                             group=mpu.get_model_parallel_group())

                    if self.args.gradient_checkpointing:
                        self.model.module.set_force_gradient_checkpointing(True)

                    input_batch = self.losses.get_input_batch(batch, lm_batch)
                    logits = self.forward_model(input_batch).logits
                    ppo_logits = logits[:batch.query_tensors.size(0)]
                    lm_logits = logits[batch.query_tensors.size(0):]

                    # forward
                    forward_time = time()
                    # compute rl-related loss on explored data
                    rl_loss, rl_loss_stats = self.losses.ppo_loss(batch, ppo_logits)
                    stats.update(rl_loss_stats)

                    # compute lm-related loss on pre-training data (optinal)
                    if self.lm_pipeline is not None:
                        pt_loss, pt_loss_stats = self.losses.pt_loss(lm_batch, lm_logits)
                        stats.update(pt_loss_stats)
                    else:
                        pt_loss = 0

                    loss = rl_loss + self.args.lm_coef * pt_loss
                    stats["tot_loss"] = loss.item()

                    forward_time = time() - forward_time

                    # backward
                    backward_time = time()
                    self.model.backward(loss)
                    self.update_temperature(training_epoch, self.losses.ld, self.args)
                    backward_time = time() - backward_time

                    # step
                    step_time = time()
                    self.model.step()
                    step_time = time() - step_time

                    if self.args.gradient_checkpointing:
                        self.model.module.set_force_gradient_checkpointing(False)

                    if self.iter_count % self.args.gradient_accumulation_steps == 0 and \
                            ((self.global_iter_count < 10000 and (self.global_iter_count % 1000 == 0)) or \
                             self.global_iter_count % self.args.save_interval == 0):
                        self.save()

                    # eval
                    if self.iter_count % self.args.gradient_accumulation_steps == 0 and \
                            ((self.global_iter_count < 1000 and (self.global_iter_count % 100 == 0)) or \
                             (self.global_iter_count % self.args.eval_interval == 0)):
                        self.evaluate()

                    elapsed_time = forward_time + backward_time + step_time

                    stats["elapsed_time"] = elapsed_time

                    for k in stats:
                        logging_stats[k] += stats[k]

                    # Logging
                    def get_log(log_stats, one_step_time):
                        keys = ["tot_loss", "rl_loss", "pt_loss", "pg_loss", "reg_loss", "reward", "rev_kl", "stu_lens",
                                "mixed_lens"]
                        prefix = "train | data_epochs {:2d}/{:2d} | inner iter: {:3d}/{:3d} | ppo epoch: {:2d}/{:2d} | global iter: {:6d}/{:6d}".format(
                            self.sampler.epochs,
                            self.args.epochs,
                            it,
                            len(self.train_dataloader),
                            ppo_epoch,
                            self.n_updates_per_batch,
                            self.global_iter_count,
                            self.total_steps
                        )
                        suffix = "| lr: {:.4e} | scale: {:6.2f} | time: {:.3f} | step time: {:.3f}".format(
                            self.scheduler.get_last_lr()[0],
                            self.opt.cur_scale if hasattr(self.opt, "cur_scale") else 0,
                            elapsed_time,
                            one_step_time
                        )
                        for key in keys:
                            prefix += "| {}: {:.4f} ".format(key, log_stats.get(key, 0))
                        return prefix + suffix

                    mid_log_step = self.args.gradient_accumulation_steps // self.args.mid_log_num
                    mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                    if self.iter_count % mid_log_step == 0:
                        print_rank(get_log(stats, 0))

                    if self.global_iter_count % self.args.log_interval == 0 and self.iter_count % self.args.gradient_accumulation_steps == 0:
                        logging_stats = {k: v / (self.args.log_interval * self.args.gradient_accumulation_steps) for
                                         k, v in logging_stats.items()}
                        log_str = get_log(logging_stats,
                                          logging_stats.get("elapsed_time", 0) * self.args.gradient_accumulation_steps)
                        print_rank("*" * 100)
                        print_rank(log_str)
                        print_rank(self.args.save)
                        print_rank("*" * 100)
                        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
                        logging_stats = {k: 0 for k in logging_stats}

                    # end
                    if (self.global_iter_count >= self.total_steps or self.sampler.epochs >= self.args.epochs):
                        if self.global_iter_count >= self.total_steps:
                            print_rank("Reached total steps {}/{}".format(self.global_iter_count, self.total_steps))
                        else:
                            print_rank("Reached data epochs {}/{}".format(self.sampler.epochs, self.args.epochs))
                        self.save()
                        results, preds, response_texts = self.evaluate_ppo()
                        if self.eval_lm_pipeline is not None:
                            eval_pt_results = self.evaluate_pt()
                            results.update(eval_pt_results)
                        self.save_evals(preds, results, response_texts)
                        return results

                    self.iter_count += 1
                    if self.iter_count % self.args.gradient_accumulation_steps == 0:
                        self.global_iter_count += 1

                self.post_backward_callback()

            self.post_epoch_callback(training_epoch)
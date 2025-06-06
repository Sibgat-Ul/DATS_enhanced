# coding=utf-8
# 2022.6.8-Changed for Annealing knowledage distillation
#      Huawei Technologies Co., Ltd. <aref.jafari@huawei.com>
# Copyright 2022 Huawei Technologies Co., Ltd.
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

import torch.nn.functional as F
import torch.nn as nn
import transformers as transformers_module
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import torch
import numpy as np
import copy
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class DynamicTemperatureScheduler():
    def __init__(
        self,
        experiment_name,
        distiller,
        train_loader,
        val_loader,
        cfg,
    ):
        super(DynamicTemperatureScheduler, self).__init__(experiment_name, distiller, train_loader, val_loader, cfg)

        self.current_temperature = cfg.INIT_TEMPERATURE
        self.initial_temperature = cfg.INIT_TEMPERATURE
        self.min_temperature = cfg.MIN_TEMPERATURE
        self.max_temperature = cfg.MAX_TEMPERATURE
        self.max_epochs = cfg.EPOCHS
        self.adjust_temp = True
        self.curve_shape = 1

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

    def get_temperature(self):
        """
        Retrieve current temperature value.

        Returns:
            float: Current dynamic temperature.
        """

        return self.current_temperature
# Create global variable to store current best result, to be used for saving best model based on dev set
BEST_ACCURACY = -1

def train(args, train_dataset, model_student, model_teacher, tokenizer, phase):
    """ Train the model """
    model_student.train()
    model_teacher.eval()


    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        if phase == 1:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs_phase_1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs_phase_2

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_student.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model_student.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.student_model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.student_model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model_student, optimizer = amp.initialize(model_student, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(model_student, torch.nn.DataParallel):
        model_student = torch.nn.DataParallel(model_student)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_student = torch.nn.parallel.DistributedDataParallel(
            model_student, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Start the training process
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))

    if phase == 1:
        logger.info("  Num Epochs = %d", args.num_train_epochs_phase_1)
    else:
        logger.info("  Num Epochs = %d", args.num_train_epochs_phase_2
                    )
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.student_model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.student_model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoc and not isinstance(model, torch.nn.DataParallel)h",
            steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model_student.zero_grad()

    set_seed(args)  # Added here for reproductibility

    if phase == 1:
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs_phase_1), desc="Epoch", disable=args.local_rank not in [-1, 0],
        )
        temp = 1
    else:
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs_phase_2), desc="Epoch", disable=args.local_rank not in [-1, 0],
        )

    for i in train_iterator:
        # If the phase is 1, then we need to increment the temperature after every (num_epochs_phase_1/max_temp) epochs
        # For e.g. num_epochs_phase_1 = 20, max_temp = 10, then we increment temperature value after every 2 epochs
        if phase == 1:
            if i % int(args.num_train_epochs_phase_1 / args.max_temperature) == 0 and i > 0:
                temp += 1
                print(f"temperature is {temp}")

        # Determine the metric based on task, then run evaluation and save model if better than current best result
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        print("Intermediate evaluate: ", evaluate(args, model_student, tokenizer))

        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model_student.train()
            model_teacher.eval()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            # Get the Student and Teacher outputs, and the groundtruth labels
            S = model_student(input_ids=batch[0], attention_mask=batch[1],
                              token_type_ids=batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None)[0]
            T = model_teacher(input_ids=batch[0], attention_mask=batch[1],
                              token_type_ids=batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None)[0]
            labels = batch[3]

            # Calculate loss based on phase
            # In Phase 1, use MSE loss between student and Annealed teacher output
            # In Phase 2, use Cross-entropy loss between student and true labels
            if phase == 1:
                loss = F.mse_loss(S, T * (temp / args.max_temperature))
                loss = F.mse_loss(S, T * (temp / args.max_temperature))
            else:
                # STS-B is a regression task, so MSE loss but other tasks are classfication
                if args.task_name == 'sts-b':
                    loss = F.mse_loss(S, labels)
                else:
                    loss = F.cross_entropy(S, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model_student.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model_student.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model_student, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model_student.module if hasattr(model_student, "module") else model_student
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def train_with_scheduler(args, train_dataset, model_student, model_teacher, tokenizer, phase):
    """ Train the model """
    model_student.train()
    model_teacher.eval()
    DTS = DynamicTemperatureScheduler()


    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        if phase == 1:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs_phase_1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs_phase_2

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_student.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model_student.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.student_model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.student_model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model_student, optimizer = amp.initialize(model_student, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(model_student, torch.nn.DataParallel):
        model_student = torch.nn.DataParallel(model_student)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_student = torch.nn.parallel.DistributedDataParallel(
            model_student, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Start the training process
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))

    if phase == 1:
        logger.info("  Num Epochs = %d", args.num_train_epochs_phase_1)
    else:
        logger.info("  Num Epochs = %d", args.num_train_epochs_phase_2
                    )
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.student_model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.student_model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoc and not isinstance(model, torch.nn.DataParallel)h",
            steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model_student.zero_grad()

    set_seed(args)  # Added here for reproductibility

    if phase == 1:
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs_phase_1), desc="Epoch", disable=args.local_rank not in [-1, 0],
        )
        temp = 1
    else:
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs_phase_2), desc="Epoch", disable=args.local_rank not in [-1, 0],
        )

    for i in train_iterator:
        # If the phase is 1, then we need to increment the temperature after every (num_epochs_phase_1/max_temp) epochs
        # For e.g. num_epochs_phase_1 = 20, max_temp = 10, then we increment temperature value after every 2 epochs
        if phase == 1:
            if i % int(args.num_train_epochs_phase_1 / args.max_temperature) == 0 and i > 0:
                temp += 1
                print(f"temperature is {temp}")

        # Determine the metric based on task, then run evaluation and save model if better than current best result
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        print("Intermediate evaluate: ", evaluate(args, model_student, tokenizer))

        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model_student.train()
            model_teacher.eval()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            # Get the Student and Teacher outputs, and the groundtruth labels
            S = model_student(input_ids=batch[0], attention_mask=batch[1],
                              token_type_ids=batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None)[0]
            T = model_teacher(input_ids=batch[0], attention_mask=batch[1],
                              token_type_ids=batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None)[0]
            labels = batch[3]

            # Calculate loss based on phase
            # In Phase 1, use MSE loss between student and Annealed teacher output
            # In Phase 2, use Cross-entropy loss between student and true labels
            if phase == 1:
                loss = F.mse_loss(S, T * (temp / args.max_temperature))
                loss = F.mse_loss(S, T * (temp / args.max_temperature))
            else:
                # STS-B is a regression task, so MSE loss but other tasks are classfication
                if args.task_name == 'sts-b':
                    loss = F.mse_loss(S, labels)
                else:
                    loss = F.cross_entropy(S, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model_student.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model_student.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model_student, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model_student.module if hasattr(model_student, "module") else model_student
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step



def evaluate(args, model, tokenizer, prefix=""):
    global BEST_ACCURACY
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model_student = torch.nn.DataParallel(model_student)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if args.task_name == 'sts-b':
        intermediate_res = results['corr']
    elif args.task_name == 'cola':
        intermediate_res = results['mcc']
    elif args.task_name == 'mrpc':
        intermediate_res = results['f1']
    elif args.task_name == 'mnli':
        intermediate_res = results['mnli/acc']
    else:
        intermediate_res = results['acc']
    print("Intermediate evaluate: ", intermediate_res)
    if BEST_ACCURACY < intermediate_res:
        BEST_ACCURACY = intermediate_res
        if args.do_train:
            torch.save(model, args.output_dir + "/BestModel.pt")
            print("Best Model Saved!")

    return results

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.student_model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

    # Roberta and DistilBERT don't use token_type ids
    if args.model_type == 'roberta' or args.model_type == 'distilbert':
        all_token_type_ids = torch.tensor([0 for f in features], dtype=torch.long)
    else:
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--init_temperature", type=float, default=8.0)
    parser.add_argument("--max_temperature", type=float, default=8.0)
    parser.add_argument("--min_temperature", type=float, default=4.0)

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--student_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained student model or shortcut name selected in the list: ",
    )

    parser.add_argument(
        "--teacher_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained teacher model or shortcut name selected in the list: ",
    )

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected from the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs_phase_1", default=10.0, type=float, help="Total number of training epochs in phase 1.",
    )

    parser.add_argument(
        "--num_train_epochs_phase_2", default=3.0, type=float, help="Total number of training epochs in phase 2.",
    )

    parser.add_argument(
        "--max_temperature", default=10, type=int, help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    # Load the Teacher model
    # First load the config and tokenizer, and finally the model
    config_teacher = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.teacher_model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer_teacher = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.teacher_model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model_teacher = transformers_module.AutoModelForSequenceClassification.from_pretrained(
        args.teacher_model_name_or_path)

    # Load the Student model
    # First load the config, then tokenizer, and finally the model
    config_student = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.student_model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer_student = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.student_model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model_student = AutoModelForSequenceClassification.from_pretrained(
        args.student_model_name_or_path,
        from_tf=bool(".ckpt" in args.student_model_name_or_path),
        config=config_student,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model_teacher.to(args.device)
    model_student.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer_student, evaluate=False)
        print("*" * 100)
        print("*" + " " * 39 + "Phase 1 of training" + " " * 40 + "*")
        print("*" * 100)

        # Train student model in phase 1
        global_step, tr_loss = train(args, train_dataset, model_student, model_teacher, tokenizer_student, phase=1)

        print("*" * 100)
        print("*" + " " * 39 + "Phase 2 of training" + " " * 38 + "*")
        print("*" * 100)

        # Now load best model saved from phase 1 to start training in phase 2
        model_student = torch.load(args.output_dir + "/BestModel.pt")

        # Train student model in phase 2
        global_step, tr_loss = train(args, train_dataset, model_student, model_teacher, tokenizer_student, phase=2)
        evaluate(args, model_student, tokenizer_student)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model_student.module if hasattr(model_student, "module") else model_student
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer_student.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

            # Load a trained model and vocabulary that you have fine-tuned
            model_student = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
            tokenizer_student = AutoTokenizer.from_pretrained(args.output_dir)
            model_student.to(args.device)

        # Evaluation
        results = {}
        if args.do_eval and args.local_rank in [-1, 0]:
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in
                    sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                model_student = AutoModelForSequenceClassification.from_pretrained(checkpoint)
                model_student.to(args.device)
                result = evaluate(args, model_student, tokenizer, prefix=prefix)
                result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                results.update(result)

    return results


if __name__ == "__main__":
    main()
    print(f"BEST PERFORMANCE IS: {BEST_ACCURACY}")



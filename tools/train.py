import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import wandb

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset, get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict

def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME

    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG

    tags = cfg.EXPERIMENT.TAG.split(",")

    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)

    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)

    if cfg.LOG.WANDB:
        try:
            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=cfg.EXPERIMENT.NAME, tags=tags)

        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    if cfg.DISTILLER.TYPE == 'MLKD':
        train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg)

    else:
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)

        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )

        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation

    else:
        print(log_msg("Loading teacher model", "INFO"))

        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)

        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]

            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)

            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )

        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )

        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )

    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )

    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")

    parser.add_argument("--exp_name", type=str, help="experiment name", default="DTAD_Experiment")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--distiller_type", type=str, default="NONE")

    parser.add_argument("--resume", type=bool, default=False)

    parser.add_argument("--use_scheduler", type=bool, default=False)
    parser.add_argument("--init_temperature", type=float, default=4.0)
    parser.add_argument("--min_temperature", type=float, default=2.0)
    parser.add_argument("--max_temperature", type=float, default=4.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)

    parser.add_argument("--logit-stand", action="store_true")
    parser.add_argument("--base-temp", type=float, default=2)
    parser.add_argument("--kd-weight", type=float, default=9)
    parser.add_argument("--num-epochs", type=int, default=100)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.dataset == "cifar100":
        cfg.SOLVER.EPOCHS = args.epochs
        cfg.SOLVER.BATCH_SIZE = args.batch_size
        cfg.SOLVER.LR_DECAY_STAGES = [int(args.epochs*0.625), int(args.epochs*0.75), int(args.epochs*0.875)]

    if args.use_scheduler:
        cfg.SOLVER.TRAINER = "scheduler"
        cfg.SOLVER.MIN_TEMPERATURE = args.min_temperature
        cfg.SOLVER.MAX_TEMPERATURE = args.max_temperature
        cfg.SOLVER.INIT_TEMPERATURE = args.init_temperature

        if args.logit_stand and cfg.DISTILLER.TYPE in ['KD', 'DKD', 'MLKD']:
            cfg.EXPERIMENT.LOGIT_STAND = True

            if cfg.DISTILLER.TYPE == 'KD':
                cfg.KD.LOSS.KD_WEIGHT = args.kd_weight
                cfg.KD.TEMPERATURE = args.init_temperature

            elif cfg.DISTILLER.TYPE == 'DKD':
                cfg.DKD.ALPHA = cfg.DKD.ALPHA * args.kd_weight
                cfg.DKD.BETA = cfg.DKD.BETA * args.kd_weight
                cfg.DKD.T = args.init_temperature

            elif cfg.DISTILLER.TYPE == 'MLKD':
                cfg.KD.LOSS.KD_WEIGHT = args.kd_weight
                cfg.KD.TEMPERATURE = args.init_temperature

    if args.logit_stand and cfg.DISTILLER.TYPE in ['KD','DKD','MLKD']:
        cfg.EXPERIMENT.LOGIT_STAND = True

        if cfg.DISTILLER.TYPE == 'KD':
            cfg.KD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.KD.TEMPERATURE = args.base_temp

        elif cfg.DISTILLER.TYPE == 'DKD':
            cfg.DKD.ALPHA = cfg.DKD.ALPHA * args.kd_weight
            cfg.DKD.BETA = cfg.DKD.BETA * args.kd_weight
            cfg.DKD.T = args.base_temp

        elif cfg.DISTILLER.TYPE == 'MLKD':
            cfg.KD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.KD.TEMPERATURE = args.base_temp

    cfg.freeze()
    main(cfg, args.resume, args.opts)

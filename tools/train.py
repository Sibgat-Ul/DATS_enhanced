import os
import torch
import wandb
import gc

import torch.backends.cudnn as cudnn

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset, get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict

import warnings

cudnn.benchmark = True

warnings.filterwarnings("ignore")

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
            wandb.login(key="9d5a8aab3348b03e43147ae4735979a983a3e7b0")
            wandb.init(project=f"{cfg.EXPERIMENT.PROJECT}_{tags[1]}->{tags[2]}", name=f"{tags[0]}: {cfg.SOLVER.TRAINER} + ls:{cfg.EXPERIMENT.LOGIT_STAND}", tags=tags)

        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    if not cfg.REUSE:
        #show_cfg(cfg)
        print(log_msg("\nTrainer: {}\n Distiller: {}".format(cfg.SOLVER.TRAINER, cfg.DISTILLER.TYPE), "INFO"))

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

            if cfg.REUSE:
                model_student_2 = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )

        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )

        else:
            if cfg.REUSE:
                distiller1 = distiller_dict[cfg.DISTILLER.TYPE](
                    model_student, model_teacher, cfg
                )

                distiller2 = distiller_dict[cfg.DISTILLER.TYPE](
                    model_student_2, model_teacher, cfg
                )

            else:
                distiller = distiller_dict[cfg.DISTILLER.TYPE](
                    model_student, model_teacher, cfg
                )

    if cfg.REUSE:
        if cfg.DISTILLER.TYPE == "MLKD":
            cfg.SOLVER.TRAINER = "ls"
            cfg.freeze()
            print(log_msg("Trainer: {}".format(cfg.SOLVER.TRAINER), "INFO"))

            distiller = torch.nn.DataParallel(distiller1.cuda())

            if cfg.DISTILLER.TYPE != "NONE":
                print(
                    log_msg(
                        "Trainer Extra parameters of {}: {}\033[0m".format(
                            cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                        ),
                        "INFO",
                    )
                )

            trainer = trainer_dict["ls"](
                experiment_name, distiller, train_loader, val_loader, cfg
            )

            trainer.train(resume=resume)

            del distiller, trainer
            torch.cuda.empty_cache()
            gc.collect()

            cfg.defrost()

            distiller = torch.nn.DataParallel(distiller2.cuda())

            if (
                    cfg.DISTILLER.STUDENT == "resnet8x4" or
                    cfg.DISTILLER.STUDENT == "wrn_16_2" or
                    (cfg.DISTILLER.TEACHER == "resnet110" and cfg.DISTILLER.STUDENT == "resnet20") or
                    (cfg.DISTILLER.TEACHER == "resnet110" and cfg.DISTILLER.STUDENT == "resnet32") or
                    cfg.DISTILLER.STUDENT == "MobileNetV2" or cfg.DISTILLER.STUDENT == "ShuffleV2" or
                    cfg.DISTILLER.STUDENT == "ShuffleV1"
            ):
                cfg.SOLVER.INIT_TEMPERATURE = 10
                cfg.SOLVER.MAX_TEMPERATURE = 10
                cfg.SOLVER.MIN_TEMPERATURE = 1
            else:
                cfg.SOLVER.INIT_TEMPERATURE = cfg.KD.TEMPERATURE * 2
                cfg.SOLVER.MAX_TEMPERATURE = cfg.SOLVER.INIT_TEMPERATURE
                cfg.SOLVER.MIN_TEMPERATURE = cfg.KD.TEMPERATURE

            cfg.SOLVER.ADJUST_TEMPERATURE = args.adjust_temperature
            cfg.SOLVER.TRAINER = "scheduler"
            cfg.freeze()

            print(log_msg("Trainer: {}".format(cfg.SOLVER.TRAINER), "INFO"))

            if cfg.DISTILLER.TYPE != "NONE":
                print(
                    log_msg(
                        "Trainer Extra parameters of {}: {}\033[0m".format(
                            cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                        ),
                        "INFO",
                    )
                )
            trainer = trainer_dict["ourls"](
                experiment_name, distiller, train_loader, val_loader, cfg
            )

            trainer.train(resume=resume)

        elif cfg.DISTILLER.TYPE == "DTKD":
            cfg.SOLVER.TRAINER = "base"
            cfg.freeze()
            print(log_msg("Trainer: {}".format(cfg.SOLVER.TRAINER), "INFO"))

            distiller = torch.nn.DataParallel(distiller1.cuda())

            if cfg.DISTILLER.TYPE != "NONE":
                print(
                    log_msg(
                        "Trainer Extra parameters of {}: {}\033[0m".format(
                            cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                        ),
                        "INFO",
                    )
                )

            trainer = trainer_dict["base"](
                experiment_name, distiller, train_loader, val_loader, cfg
            )
            trainer.train(resume=resume)

            del distiller, trainer
            torch.cuda.empty_cache()
            gc.collect()

            cfg.defrost()

            if cfg.DISTILLER.STUDENT == "MobileNetV2" or cfg.DISTILLER.STUDENT == "ShuffleV2" or cfg.DISTILLER.STUDENT == "ShuffleV1":
                cfg.SOLVER.INIT_TEMPERATURE = 2
                cfg.SOLVER.MAX_TEMPERATURE = 3
                cfg.SOLVER.MIN_TEMPERATURE = 1
            elif cfg.DISTILLER.STUDENT == "resnet8x4" or cfg.DISTILLER.STUDENT == "wrn_16_2":
                cfg.SOLVER.INIT_TEMPERATURE = 2
                cfg.SOLVER.MAX_TEMPERATURE = 3
                cfg.SOLVER.MIN_TEMPERATURE = 1
            else:
                cfg.SOLVER.INIT_TEMPERATURE = cfg.KD.TEMPERATURE * 2
                cfg.SOLVER.MAX_TEMPERATURE = cfg.SOLVER.INIT_TEMPERATURE
                cfg.SOLVER.MIN_TEMPERATURE = cfg.KD.TEMPERATURE

            cfg.SOLVER.ADJUST_TEMPERATURE = args.adjust_temperature
            cfg.SOLVER.TRAINER = "scheduler"
            cfg.DISTILLER.TYPE = "kd"

            cfg.freeze()

            distiller = torch.nn.DataParallel(distiller2.cuda())

            print(log_msg("Trainer: {}".format(cfg.SOLVER.TRAINER), "INFO"))

            if cfg.DISTILLER.TYPE != "NONE":
                print(
                    log_msg(
                        "Trainer Extra parameters of {}: {}\033[0m".format(
                            cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                        ),
                        "INFO",
                    )
                )
            trainer = trainer_dict["scheduler"](
                experiment_name, distiller, train_loader, val_loader, cfg
            )

            trainer.train(resume=resume)

        else:
            cfg[cfg.DISTILLER.TYPE].WARMUP = 5
            cfg.SOLVER.TRAINER = "base"
            cfg.freeze()

            print(log_msg("Trainer: {}".format(cfg.SOLVER.TRAINER), "INFO"))

            distiller = torch.nn.DataParallel(distiller1.cuda())

            if cfg.DISTILLER.TYPE != "NONE":
                print(
                    log_msg(
                        "Trainer Extra parameters of {}: {}\033[0m".format(
                            cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                        ),
                        "INFO",
                    )
                )

            trainer = trainer_dict["base"](
                experiment_name, distiller, train_loader, val_loader, cfg
            )

            trainer.train(resume=resume)

            del distiller, trainer
            torch.cuda.empty_cache()
            gc.collect()

            cfg.defrost()

            distiller = torch.nn.DataParallel(distiller2.cuda())
            
            if cfg.DISTILLER.STUDENT == "MobileNetV2" or cfg.DISTILLER.STUDENT == "ShuffleV2":
                cfg.SOLVER.INIT_TEMPERATURE = 3
                cfg.SOLVER.MAX_TEMPERATURE = 4
                cfg.SOLVER.MIN_TEMPERATURE = 2
            elif cfg.DISTILLER.STUDENT == "resnet8x4" or cfg.DISTILLER.STUDENT == "wrn_16_2":
                cfg.SOLVER.INIT_TEMPERATURE = 2
                cfg.SOLVER.MAX_TEMPERATURE = 3
                cfg.SOLVER.MIN_TEMPERATURE = 1
            else:
                cfg.SOLVER.INIT_TEMPERATURE = cfg[cfg.DISTILLER.TYPE].TEMPERATURE * 2
                cfg.SOLVER.MAX_TEMPERATURE = cfg.SOLVER.INIT_TEMPERATURE
                cfg.SOLVER.MIN_TEMPERATURE = cfg[cfg.DISTILLER.TYPE].TEMPERATURE

            cfg.SOLVER.ADJUST_TEMPERATURE = args.adjust_temperature
            cfg.SOLVER.TRAINER = "scheduler"
            cfg.freeze()

            print(log_msg("Trainer: {}".format(cfg.SOLVER.TRAINER), "INFO"))

            if cfg.DISTILLER.TYPE != "NONE":
                print(
                    log_msg(
                        "Extra parameters of {}: {}\033[0m".format(
                            cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                        ),
                        "INFO",
                    )
                )
            trainer = trainer_dict["scheduler"](
                experiment_name, distiller, train_loader, val_loader, cfg
            )

            trainer.train(resume=resume)

    else:
        # train
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

    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--init_temperature", type=float, default=4.0)
    parser.add_argument("--min_temperature", type=float, default=2.0)
    parser.add_argument("--max_temperature", type=float, default=4.0)
    parser.add_argument("--adjust_temperature", action="store_true")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)

    parser.add_argument("--logit_stand", action="store_true")
    parser.add_argument("--base_temp", type=float, default=2)
    parser.add_argument("--kd_weight", type=float, default=9)
    parser.add_argument("--num_epochs", type=int, default=100)

    parser.add_argument("--reuse", action="store_true")
    # parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--curve_shape", type=int, default=1)

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
        cfg.SOLVER.ADJUST_TEMPERATURE = args.adjust_temperature

        if cfg.DISTILLER.STUDENT == "resnet8x4" or cfg.DISTILLER.STUDENT == "wrn_16_2":
            cfg.SOLVER.INIT_TEMPERATURE = 2
            cfg.SOLVER.MAX_TEMPERATURE = 3
            cfg.SOLVER.MIN_TEMPERATURE = 1

    if args.distiller_type == "MLKD":
        cfg.SOLVER.TRAINER = "ls"

    cfg.EXPERIMENT.LOGIT_STAND = args.logit_stand

    if args.logit_stand is True and cfg.DISTILLER.TYPE in ['KD','DKD','MLKD']:
        cfg.EXPERIMENT.LOGIT_STAND = True

        if cfg.DISTILLER.TYPE == 'KD':
            cfg.KD.LOSS.KD_WEIGHT = 9.0
            cfg.KD.TEMPERATURE = args.base_temp

        elif cfg.DISTILLER.TYPE == 'DKD':
            cfg.DKD.ALPHA = 1.0 * args.kd_weight
            cfg.DKD.BETA = 8.0 * args.kd_weight
            cfg.DKD.T = args.base_temp

        elif cfg.DISTILLER.TYPE == 'MLKD':
            cfg.KD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.KD.TEMPERATURE = args.base_temp

    cfg.LOG.WANDB = args.wandb
    cfg.REUSE = True if args.reuse else False

    if not cfg.REUSE:
        cfg.freeze()

    main(cfg, args.resume, args.opts)

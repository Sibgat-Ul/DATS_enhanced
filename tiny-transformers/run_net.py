#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/tools/run_net.py
"""

import argparse
import sys
import os

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
from pycls.core.config import cfg
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a model.")
    help_s, choices = "Run mode", ["train", "test", "time"]
    parser.add_argument("--mode", help=help_s, choices=choices, required=True, type=str)
    help_s = "Config file location"
    parser.add_argument("--epochs", help=help_s, default=100, type=int)
    parser.add_argument("--use_scheduler", help=help_s, action="store_true")
    parser.add_argument("--use_inter", help=help_s, action="store_true")

    parser.add_argument("--inter_weight", help=help_s, type=float, default=2.5)
    parser.add_argument("--kd_weight", help=help_s, type=float, default=0.5)
    parser.add_argument("--extra_kd_weight", help=help_s, type=float, default=2.5)

    parser.add_argument("--min_temp", type=float, help=help_s, default=2)
    parser.add_argument("--max_temp", type=float, help=help_s, default=6)
    parser.add_argument("--init_temp", type=float, help=help_s, default=6)
    parser.add_argument("--logit_stand", action="store_true")
    parser.add_argument("--curve_shape", type=float, default=1.0)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--cfg", help=help_s, required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    config.load_cfg(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.DISTILLATION.SCHEDULE = args.use_scheduler
    cfg.DISTILLATION.LOGIT_STANDARD = args.logit_stand
    cfg.DISTILLATION.INTER_WEIGHT = args.inter_weight
    cfg.DISTILLATION.LOGIT_WEIGHT = args.kd_weight
    cfg.DISTILLATION.EXTRA_WEIGHT_IN = args.extra_kd_weight

    cfg.OPTIM.MAX_EPOCH = args.epochs
    cfg.NUM_GPUS = args.n_gpu

    cfg.TEMPERATURE.MIN = args.min_temp
    cfg.TEMPERATURE.MAX = args.max_temp
    cfg.TEMPERATURE.INIT = args.init_temp

    cfg.DISTILLATION.CURVE_SHAPE = args.curve_shape
    cfg.DISTILLATION.ENABLE_INTER = args.use_inter

    if cfg.OUT_DIR is None:
        out_dir = os.path.join('work_dirs', os.path.splitext(os.path.basename(args.cfg))[0])
        cfg.OUT_DIR = out_dir
    config.assert_cfg()
    cfg.freeze()
    if not os.path.exists(cfg.OUT_DIR):
        os.makedirs(cfg.OUT_DIR)
    if mode == "train":
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_model)
    elif mode == "test":
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.test_model)
    elif mode == "time":
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.time_model)


if __name__ == "__main__":
    main()

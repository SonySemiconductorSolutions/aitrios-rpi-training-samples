# Copyright 2021 RangiLyu.
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

import torch
import pytorch_lightning as pl

import argparse

from third_party.mct.nanodet_keras_model import set_nanodet_classes

try:  # for setup_pretrained()
    from nanodet.util import (
        NanoDetLightningLogger,
        cfg,
        load_config,
        mkdir,
    )
except ImportError as e:  # package not installed, skip
    print('  Warning : nanodet is not installed, "pip install ."')


class NanodetPlus:
    def __init__(self, config):
        # call from imx500_zoo.py/Solution.setup_model()
        self.config = config

    def setup_retrain(self):
        args = self._setup_args()
        load_config(cfg, args.config)
        self._setup_cfg(cfg)
        self.config.nanodet.cfg = cfg

        if cfg.model.arch.head.num_classes != len(cfg.class_names):
            raise ValueError(
                "cfg.model.arch.head.num_classes must equal len(cfg.class_names), "
                "but got {} and {}".format(
                    cfg.model.arch.head.num_classes, len(cfg.class_names)
                )
            )
        self._set_classes_quantizer()

        local_rank = int(args.local_rank)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        mkdir(local_rank, cfg.save_dir)

        logger = NanoDetLightningLogger(cfg.save_dir)
        self.config.nanodet.logger = logger
        logger.dump_cfg(cfg)

        if args.seed is not None:
            logger.info("Set random seed to {}".format(args.seed))
            pl.seed_everything(args.seed)

    def _setup_cfg(self, cfg):
        cfg.defrost()

        t = self.config["TRAINER"]
        cfg.schedule.total_epochs = int(t["NUM_EPOCHS"])
        cfg.device.batchsize_per_gpu = int(t["BATCH_SIZE"])
        cfg.device.workers_per_gpu = int(t["NUM_WORKERS"])

        nclass = len(cfg.class_names)
        cfg.model.arch.head.num_classes = nclass
        cfg.model.arch.aux_head.num_classes = nclass
        self.config["MODEL"]["CLASS_NUM"] = f"{nclass}"

        nchannel = int(self.config["MODEL"]["FEATURE_CHANNELS"])
        cfg.model.arch.head.input_channel = nchannel
        cfg.model.arch.head.feat_channels = nchannel
        cfg.model.arch.fpn.out_channels = nchannel
        cfg.model.arch.aux_head.input_channel = nchannel * 2
        cfg.model.arch.aux_head.input_channel = nchannel * 2

        cfg.schedule.warmup.ratio = float(t["LEARNING_RATE"])

        cfg.freeze()

    def _set_classes_quantizer(self, num=-1, is_init=False):
        if is_init:
            set_nanodet_classes()
        else:
            n = len(self.config.nanodet.cfg.class_names) if num == -1 else num
            set_nanodet_classes(n)

    def _setup_args(self):
        args = argparse.ArgumentParser()
        args.add_argument("config", help="train config file path")
        args.add_argument(
            "--local_rank",
            default=-1,
            type=int,
            help="node rank for distributed training",
        )
        args.add_argument("--seed", type=int, default=None, help="random seed")

        args.local_rank = -1
        args.seed = None
        args.config = self.config["TRAINER"]["CONFIG"]

        self.config.nanodet.args = args
        return args

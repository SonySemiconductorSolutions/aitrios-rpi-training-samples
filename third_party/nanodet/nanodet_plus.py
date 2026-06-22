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
    print(f'  Warning : nanodet is not installed, "pip install ." : {e}')


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
        
        # Training Parameters
        t = self.config["TRAINER"]
        cfg.schedule.total_epochs = int(t["NUM_EPOCHS"])
        cfg.device.batchsize_per_gpu = int(t["BATCH_SIZE"])
        cfg.device.workers_per_gpu = int(t["NUM_WORKERS"])
        
        # Model Architecture
        m = self.config["MODEL"]
        
        # Backbone
        if "BACKBONE_SIZE" in m:
            cfg.model.arch.backbone.model_size = m["BACKBONE_SIZE"]
        
        if "BACKBONE_ACTIVATION" in m:
            activation = m["BACKBONE_ACTIVATION"]
            cfg.model.arch.backbone.activation = activation
            cfg.model.arch.fpn.activation = activation
            cfg.model.arch.head.activation = activation
            cfg.model.arch.aux_head.activation = activation
        
        # FPN
        if "FPN_KERNEL_SIZE" in m:
            kernel = int(m["FPN_KERNEL_SIZE"])
            cfg.model.arch.fpn.kernel_size = kernel
            cfg.model.arch.head.kernel_size = kernel
        
        if "FPN_USE_DEPTHWISE" in m:
            cfg.model.arch.fpn.use_depthwise = (m["FPN_USE_DEPTHWISE"].lower() == "true")
        
        if "FPN_NUM_EXTRA_LEVEL" in m:
            cfg.model.arch.fpn.num_extra_level = int(m["FPN_NUM_EXTRA_LEVEL"])
        
        # Classes and Channels
        nclass = len(cfg.class_names)
        cfg.model.arch.head.num_classes = nclass
        cfg.model.arch.aux_head.num_classes = nclass
        self.config["MODEL"]["CLASS_NUM"] = f"{nclass}"
        
        nchannel = int(m["FEATURE_CHANNELS"])
        cfg.model.arch.head.input_channel = nchannel
        cfg.model.arch.head.feat_channels = nchannel
        cfg.model.arch.fpn.out_channels = nchannel
        cfg.model.arch.aux_head.input_channel = nchannel * 2
        cfg.model.arch.aux_head.feat_channels = nchannel * 2
        
        # Learning Rate
        cfg.schedule.warmup.ratio = float(t["LEARNING_RATE"])
        
        # Loss Functions
        if "LOSS" in self.config:
            loss = self.config["LOSS"]
            
            if "QFL_BETA" in loss:
                cfg.model.arch.head.loss.loss_qfl.beta = float(loss["QFL_BETA"])
            
            if "QFL_WEIGHT" in loss:
                cfg.model.arch.head.loss.loss_qfl.loss_weight = float(loss["QFL_WEIGHT"])
            
            if "DFL_WEIGHT" in loss:
                cfg.model.arch.head.loss.loss_dfl.loss_weight = float(loss["DFL_WEIGHT"])
            
            if "BBOX_LOSS_TYPE" in loss:
                cfg.model.arch.head.loss.loss_bbox.name = loss["BBOX_LOSS_TYPE"]
            
            if "BBOX_LOSS_WEIGHT" in loss:
                cfg.model.arch.head.loss.loss_bbox.loss_weight = float(loss["BBOX_LOSS_WEIGHT"])
        
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


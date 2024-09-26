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
import os
import copy

from third_party.nanodet.coco_a import CocoADataset

import albumentations as albu

try:  # for setup_retrain()
    from nanodet.data.collate import naive_collate
    from nanodet.evaluator import build_evaluator
except ImportError as e:  # package not installed, skip
    print('  Warning : nanodet is not installed, "pip install ."')


class CardDetection:
    def __init__(self, config):
        self.config = config

    def setup_retrain(self):
        logger = self.config.nanodet.logger
        logger.info("Setting up data...")
        cfg = self.config.nanodet.cfg

        train_dataset = self._build_dataset(self._get_dir(cfg, True), "train")
        val_dataset = self._build_dataset(self._get_dir(cfg, False), "test")

        transforms = self._setup_augmentation()
        train_dataset.transform = transforms["train_a"]

        evaluator = build_evaluator(cfg.evaluator, val_dataset)
        self.config.nanodet.evaluator = evaluator

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.device.batchsize_per_gpu,
            shuffle=True,
            num_workers=cfg.device.workers_per_gpu,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=True,
        )
        self.trainloader = train_dataloader

        torch.manual_seed(
            0xBEEF
        )  # fix shuffle of val_dataloader with pseudo random number
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.device.batchsize_per_gpu,
            shuffle=True,
            num_workers=cfg.device.workers_per_gpu,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=False,
        )
        self.validloader = val_dataloader

    def _build_dataset(self, cfg, mode):
        dataset_cfg = copy.deepcopy(cfg)
        name = dataset_cfg.pop("name")
        return CocoADataset(mode=mode, **dataset_cfg)

    def _get_dir(self, cfg, is_train=True):
        if is_train:
            dir = copy.deepcopy(cfg.data.train)
            js = "annotations_train.json"
            pt = self.train_path
        else:
            dir = copy.deepcopy(cfg.data.val)
            js = "annotations_val.json"
            pt = self.valid_path

        dir["img_path"] = pt
        dir["ann_path"] = os.path.join(self.annotations_path, js)
        return dir

    def _setup_augmentation(self):
        return {
            "train_a": albu.Compose(
                [
                    albu.HorizontalFlip(p=0.5),
                    albu.VerticalFlip(p=0.5),
                    albu.Rotate(limit=(-60, 60), p=0.5),
                    albu.ShiftScaleRotate(scale_limit=(0, 0), p=0.5),
                    albu.HueSaturationValue(p=0.5),
                    albu.ElasticTransform(p=0.5),
                ],
                bbox_params=albu.BboxParams(
                    format="pascal_voc", label_fields=["class_labels"]
                ),  #            bbox_params=albu.BboxParams(format="coco", label_fields=["class_labels"])
            ),
            "valid_a": albu.Compose(
                [
                    albu.Normalize(),
                ],
                bbox_params=albu.BboxParams(
                    format="coco", label_fields=["class_labels"]
                ),
            ),
        }

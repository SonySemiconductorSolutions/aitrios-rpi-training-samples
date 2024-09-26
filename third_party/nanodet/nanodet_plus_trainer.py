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

import os
import warnings
import json

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
import torch.distributed as dist
import pickle

from imx500_zoo import utilities

try:
    from nanodet.trainer.task import TrainingTask
    from nanodet.util import (
        convert_old_model,
        env_utils,
        load_model_weight,
    )
    from nanodet.util import mkdir
except ImportError as e:  # package not installed, skip
    print('  Warning : nanodet is not installed, "pip install ."')

DOWNLOAD_CHECKPOINT = "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m-1.5x_416_checkpoint.ckpt"


class NanodetPlusTrainer:
    def __init__(self, config):
        # call from imx500_zoo.py/Solution.setup_trainer(), setup_model()/setup_data() are executed
        self.config = config

    def _setup(self):
        logger = self.config.nanodet.logger
        nanodet = self.config.nanodet
        cfg = nanodet.cfg
        logger.info("Creating model...")
        self.task = TrainingTaskExpand(cfg, nanodet.evaluator)

        is_cpu = not torch.cuda.is_available()
        if "load_model" in cfg.schedule:
            fmodel = os.path.join("./", cfg.schedule.load_model)
            utilities.download_file(
                DOWNLOAD_CHECKPOINT,
                os.path.dirname(fmodel),
            )
            ml = torch.device("cpu") if is_cpu else None
            ckpt = torch.load(fmodel, map_location=ml)
            if "pytorch-lightning_version" not in ckpt:
                warnings.warn(
                    "Warning! Old .pth checkpoint is deprecated. "
                    "Convert the checkpoint with tools/convert_old_checkpoint.py "
                )
                ckpt = convert_old_model(ckpt)
            load_model_weight(self.task.model, ckpt, logger)
            logger.info("Loaded model weight from {}".format(fmodel))

        self.model_resume_path = (
            os.path.join(cfg.save_dir, "model_last.ckpt")
            if "resume" in cfg.schedule
            else None
        )
        if is_cpu:
            logger.info("Using CPU training")

            accelerator, devices, strategy, precision = (
                "cpu",
                "auto",
                "auto",
                cfg.device.precision,
            )
        else:
            accelerator, devices, strategy, precision = (
                "gpu",
                cfg.device.gpu_ids,
                "auto",
                cfg.device.precision,
            )

        if devices and len(devices) > 1:
            strategy = "ddp"
            env_utils.set_multi_processing(distributed=True)

        self.trainer = pl.Trainer(
            default_root_dir=cfg.save_dir,
            max_epochs=cfg.schedule.total_epochs,
            check_val_every_n_epoch=cfg.schedule.val_intervals,
            accelerator=accelerator,
            devices=devices,
            log_every_n_steps=cfg.log.interval,
            num_sanity_val_steps=0,
            callbacks=[TQDMProgressBar(refresh_rate=0)],  # disable tqdm bar
            logger=logger,
            benchmark=cfg.get("cudnn_benchmark", True),
            gradient_clip_val=cfg.get("grad_clip", 0.0),
            strategy=strategy,
            precision=precision,
        )

    def fit(self, model, dataloader_train, dataloader_valid):
        self._setup()
        cfg = self.config.nanodet.cfg
        logger = self.config.nanodet.logger

        classes = cfg.class_names
        logger.info(f"  classes : {len(classes)} - {classes}")
        self.trainer.fit(
            self.task,
            dataloader_train,
            dataloader_valid,
            ckpt_path=self.model_resume_path,
        )

        model.setup_pretrained(  # for NonMaximumSupression
            cfg.save_dir, "model_best/nanodet_model_best.pth"
        )


class TrainingTaskExpand(TrainingTask):  # nanodet\trainer\task.py
    def on_validation_epoch_end(self):  # override to modify gather_results()
        results = {}
        for res in self.validation_step_outputs:
            results.update(res)
        all_results = (
            self.gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            eval_results = self.evaluator.evaluate(
                all_results, self.cfg.save_dir, rank=self.local_rank
            )
            metric = eval_results[self.cfg.evaluator.save_key]
            # save best model
            if metric > self.save_flag:
                self.save_flag = metric
                best_save_path = os.path.join(self.cfg.save_dir, "model_best")
                mkdir(self.local_rank, best_save_path)
                self.trainer.save_checkpoint(
                    os.path.join(best_save_path, "model_best.ckpt")
                )
                self.save_model_state(
                    os.path.join(best_save_path, "nanodet_model_best.pth")
                )
                txt_path = os.path.join(best_save_path, "eval_results.txt")
                if self.local_rank < 1:
                    with open(txt_path, "a") as f:
                        f.write("Epoch:{}\n".format(self.current_epoch + 1))
                        for k, v in eval_results.items():
                            f.write("{}: {}\n".format(k, v))
            else:
                warnings.warn(
                    "Warning! Save_key is not in eval results! Only save model last!"
                )
            self.logger.log_metrics(eval_results, self.current_epoch + 1)
        else:
            self.logger.info("Skip val on rank {}".format(self.local_rank))

        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):  # override to modify gather_results()
        results = {}
        for res in self.test_step_outputs:
            results.update(res)
        all_results = (
            self.gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            res_json = self.evaluator.results2json(all_results)
            json_path = os.path.join(self.cfg.save_dir, "results.json")
            json.dump(res_json, open(json_path, "w"))

            if self.cfg.test_mode == "val":
                eval_results = self.evaluator.evaluate(
                    all_results, self.cfg.save_dir, rank=self.local_rank
                )
                txt_path = os.path.join(self.cfg.save_dir, "eval_results.txt")
                with open(txt_path, "a") as f:
                    for k, v in eval_results.items():
                        f.write("{}: {}\n".format(k, v))
        else:
            self.logger.info("Skip test on rank {}".format(self.local_rank))
        self.test_step_outputs.clear()  # free memory

    def gather_results(  # modify to use "cpu", nanodet\util\scatter_gather.py
        self, result_part
    ):
        rank = -1
        world_size = 1
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        # dump result part to tensor with pickle
        part_tensor = torch.tensor(
            bytearray(pickle.dumps(result_part)),
            dtype=torch.uint8,
            device=dev,
        )

        # gather all result part tensor shape
        shape_tensor = torch.tensor(part_tensor.shape, device=dev)
        shape_list = [shape_tensor.clone() for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)

        # padding result part tensor to max length
        shape_max = torch.tensor(shape_list).max()
        part_send = torch.zeros(shape_max, dtype=torch.uint8, device=dev)
        part_send[: shape_tensor[0]] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max) for _ in range(world_size)
        ]

        # gather all result dict
        dist.all_gather(part_recv_list, part_send)

        if rank < 1:
            all_res = {}
            for recv, shape in zip(part_recv_list, shape_list):
                all_res.update(
                    pickle.loads(recv[: shape[0]].cpu().numpy().tobytes())
                )
            return all_res

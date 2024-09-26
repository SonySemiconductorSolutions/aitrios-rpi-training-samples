# coding: utf-8

import sys, os
import re
import datetime

import configparser
import torch

torch.set_float32_matmul_precision("medium")

import imx500_zoo.models
import imx500_zoo.datasets
import imx500_zoo.trainers
import imx500_zoo.quantizers
import imx500_zoo.utilities
import imx500_zoo.validators


class Solution:
    def __init__(self, config_file_path="config.ini"):
        # load configuration
        self.config_file_path = config_file_path
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file_path, encoding="utf-8")
        self.solution_name = os.path.basename(
            self.config_file_path
        )  # basename
        self.solution_name = os.path.splitext(self.solution_name)[
            0
        ]  # without extention
        self.config["SOLUTION"]["NAME"] = self.solution_name
        self.start_date = datetime.datetime.today()

        self.update_config(self.config)

    # config
    def update_config(self, config):
        self.config = config
        self._parse_config()

    # setup
    def setup(self):
        print("[IMX500_zoo] Solution setting")
        self.retrain_enable = self.config["SOLUTION"]["RETRAIN"]
        self.validate_enable = self.config["SOLUTION"]["VALIDATE"]

        self._setup_target_pathes()
        self._setup_model()
        self._setup_data()
        self._setup_trainer()
        self._setup_quantizer()
        self._setup_validator()

    def _setup_validator(self):
        if self.config.is_validate:
            validator_name = self.config["VALIDATOR"]["NAME"]
            self.validator = self._setup_module(
                validator_name, imx500_zoo.validators
            )
        else:
            self.validator = None

    def _setup_quantizer(self):
        quantizer_name = self.config["QUANTIZER"]["NAME"]
        self.quantizer = self._setup_module(
            quantizer_name, imx500_zoo.quantizers
        )

    def _setup_trainer(self):
        if self.retrain_enable == "True":
            trainer_name = self.config["TRAINER"]["NAME"]
            self.trainer = self._setup_module(
                trainer_name, imx500_zoo.trainers
            )
        else:
            self.trainer = None

    def _setup_data(self):
        data_name = self.config["DATASET"]["NAME"]

        # dataset inside imx500_zoo
        self.dataset = self._setup_module(data_name, imx500_zoo.datasets)
        if self.dataset == None:  # can not found
            print("Can't find the dataset: ", data_name)
            exit()
        self.update_data(self.dataset)

    def update_data(self, dataset):
        self.dataset = dataset
        self.dataset.setup()

        rets = self.dataset.get_loaders()
        self.dataloader_train = rets[0]
        self.dataloader_valid = rets[1]
        self.dataloader_quant = rets[2]
        self.dataloader_eval = rets[3] if len(rets) >= 4 else rets[2]

    def _setup_model(self):
        model_name = self.config["MODEL"]["NAME"]
        self.model = self._setup_module(model_name, imx500_zoo.models)
        self.model.setup()

    def _setup_module(self, module_name, modules):
        ret = None
        for m in [modules]:
            if None != getattr(m, module_name, None):
                ret = getattr(m, module_name)(self.config)
                break
        return ret

    # create
    def train(self):
        print("[IMX500_zoo] Training")
        target_folder = self.config["PATH"]["MODEL"]
        os.makedirs(target_folder, exist_ok=True)

        if self.retrain_enable == "True":
            self.trainer.fit(
                self.model, self.dataloader_train, self.dataloader_valid
            )
            if self.framework == "pytorch":
                    self.model.export_onnx(self.config["PATH"]["ONNX"])
            elif self.framework == "keras":
                self.model.export_keras(self.config["PATH"]["KERAS"])
        else:
            if self.framework == "pytorch":
                self.model.export_onnx(self.config["PATH"]["ONNX"])
            elif self.framework == "keras":
                self.model.export_keras(self.config["PATH"]["KERAS"])
            print("Skip Retraining")

    def quantize(self):
        print("[IMX500_zoo] Quantize")
        self.quantizer.quantize(
            self.model.get_trained_model(), self.dataloader_quant
        )

    def validate(self):
        print("[IMX500_zoo] Validate")
        if self.validate_enable == "True":
            validator = self.validator
            return validator.validate(self.dataloader_eval)
        else:
            print("Skip Validate")
            return 0, 0

    def _setup_target_pathes(self):
        self.config["PATH"] = {}

        conf = self.config["SOLUTION"]
        self.framework = self.config["SOLUTION"]["FRAMEWORK"]

        # ROOT PATHES
        self.config["PATH"]["MODEL_ROOT"] = conf.get(
            "MODEL_PATH", "./model"
        )  # default path is ./model
        self.config["PATH"]["DATA_ROOT"] = conf.get(
            "DATA_PATH", "./data"
        )  # default path is ./data
        self.config["PATH"]["LOG"] = conf.get(
            "LOG_PATH", "./log"
        )  # default path is ./log

        # MODEL PATHES
        model_name = self.config["SOLUTION"]["NAME"]

        target_folder = (
            self.config["PATH"]["MODEL_ROOT"] + "/" + model_name + "/"
        )
        self.config["PATH"]["MODEL"] = target_folder

        if self.framework == "pytorch":
            self.config["PATH"]["ONNX"] = target_folder + model_name + ".onnx"
            self.config["PATH"]["ONNX_SUMMARY"] = (
                target_folder + model_name + "_onnx_summary.txt"
            )

            self.config["PATH"]["QUANTIZED_ONNX"] = (
                target_folder + model_name + "_quantized.onnx"
            )
            self.config["PATH"]["QUANTIZED_ONNX_SUMMARY"] = (
                target_folder + model_name + "quantized_onnx_summary.txt"
            )

        elif self.framework == "keras":
            self.config["PATH"]["KERAS"] = (
                target_folder + model_name + ".keras"
            )
            self.config["PATH"]["KERAS_SUMMARY"] = (
                target_folder + model_name + "_keras_summary.txt"
            )

            self.config["PATH"]["QUANTIZED_KERAS"] = (
                target_folder + model_name + "_quantized.keras"
            )
            self.config["PATH"]["QUANTIZED_KERAS_SUMMARY"] = (
                target_folder + model_name + "_quantized_keras_summary.txt"
            )

        # DATASET PATHES
        data_name = self.config["DATASET"]["NAME"]
        self.config["PATH"]["DATA"] = (
            self.config["PATH"]["DATA_ROOT"] + "/" + data_name
        )

    def _parse_config(self):
        self.retrain_enable = self.config["SOLUTION"]["RETRAIN"]
        self.validate_enable = self.config["SOLUTION"]["VALIDATE"]
        self.framework = self.config["SOLUTION"]["FRAMEWORK"]

        self.config.is_retrain = self._config_value(
            "SOLUTION", "RETRAIN", is_bool=True
        )
        self.config.is_validate = self._config_value(
            "SOLUTION", "VALIDATE", is_bool=True
        )

    def _config_value(self, y=None, x=None, is_lower=False, is_bool=False):
        try:
            value = self.config[y][x]
        except Exception as e:
            print(
                f"  Warning no value : {self.config_file_path}, {y}, {x}\r\n    {e}",
                file=sys.stderr,
            )
            return None

        if is_lower:
            value = value.lower()
        if is_bool:
            value = self._tobool(value)
        return value

    def _tobool(self, str, true="true"):
        return str.lower().__contains__(true.lower())
    
    def _exist_quantized_model(self):
        if self.framework == "pytorch":
            is_exist = os.path.exists(self.config["PATH"]["QUANTIZED_ONNX"])
        elif self.framework == "keras":
            is_exist = os.path.exists(self.config["PATH"]["QUANTIZED_KERAS"])
        
        return is_exist

def main_imx500_zoo():
    if len(sys.argv) == 2 or 3:
        config_file_path = sys.argv[1]
        if len(sys.argv) == 3:
            dump_file_name = sys.argv[2]
        else:
            dump_file_name = None
        main_cli(config_file_path, dump_file_name)
    else:
        print("Usage: imx500_zoo [config.ini]")
        exit()

def main_cli(config_file_path=None, dump_file_name=None):
    config = config_file_path
        
    solution = Solution(config)
    solution.setup()
    solution.train()
    solution.quantize()
    validate_results = solution.validate()

    if dump_file_name is not None:
        imx500_zoo.utilities.dump_json(
            dump_file_name,
            solution, 
            validate_results
        )

if __name__ == "__main__":
    main_cli("nanodet_plus_as_is.ini")

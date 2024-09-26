import tensorflow as tf
from keras.models import Model
import torch

import os

from imx500_zoo import utilities

from third_party.mct.nanodet_keras_model import (
    nanodet_plus_m,
    nanodet_box_decoding,
    set_nanodet_classes,
)
from third_party.mct.torch2keras_weights_translation import load_state_dict

import third_party.nanodet.nanodet_plus as NN

DOWNLOAD_PTH = r"https://drive.google.com/u/0/uc?id=16FJJJgUt5VrSKG7RM_ImdKKzhJ-Mu45I&export=download%22"  # nanodet-plus-m-1.5x_416.pth


class NanodetPlus:
    def __init__(self, config):
        # call from imx500_zoo.py/Solution.setup_model()
        if not hasattr(config, "nanodet"):
            config.nanodet = empty_class()
        self.config = config

        self.n = NN.NanodetPlus(config)

    def _download_weights(self, pretrained_weights_path):
        utilities.download_file(
            DOWNLOAD_PTH,
            pretrained_weights_path,
            "pretrained pth file",
            "nanodet-plus-m-1.5x_416.pth",
        )

    def setup(self):
        # call from imx500_zoo.py/Solution.setup_model()
        if self.config.is_retrain:
            self._setup_retrain()
        else:
            self.setup_pretrained(is_dl=True)
                
    def _setup_retrain(self):
        self.n.setup_retrain()

    def setup_pretrained(
        self,
        pth_path="./pretrained_weights",
        pth_file="nanodet-plus-m-1.5x_416.pth",
        is_dl=False,
    ):
        input_size = int(self.config["MODEL"]["INPUT_SIZE"])
        input_shape = (input_size, input_size, 3)
        scale_factor = float(self.config["MODEL"]["SCALE_FACTOR"])
        bottleneck_ratio = float(self.config["MODEL"]["BOTTLENECK_RATIO"])
        feature_channels = int(self.config["MODEL"]["FEATURE_CHANNELS"])

        self.pretrained_weights_path = pth_path
        if is_dl:
            self._download_weights(self.pretrained_weights_path)

        pretrained_weights_file = os.path.join(
            self.pretrained_weights_path, pth_file
        )
        self._setup_class_num()
        pretrained_weights = torch.load(
            pretrained_weights_file, map_location=torch.device("cpu")
        )["state_dict"]

        # Generate Nanodet base model
        self.keras_model = nanodet_plus_m(
            input_shape, scale_factor, bottleneck_ratio, feature_channels
        )

        # Set the pre-trained weights
        load_state_dict(self.keras_model, state_dict_torch=pretrained_weights)

        # Add Nanodet Box decoding layer (decode the model outputs to bounding box coordinates)
        scores, boxes = nanodet_box_decoding(
            self.keras_model.output, res=input_size
        )

        # Add Tensorflow NMS layer
        self.outputs = tf.image.combined_non_max_suppression(
            boxes,
            scores,
            max_output_size_per_class=300,
            max_total_size=300,
            iou_threshold=0.65,
            score_threshold=0.001,
            pad_per_class=False,
            clip_boxes=False,
        )

    def _setup_class_num(self):
        try:
            num = self.config["MODEL"]["CLASS_NUM"]
            set_nanodet_classes(int(num))
        except Exception as e:
            print(f"  Warning : MODEL.CLASS_NUM is not set, default 80")

    def get(self):
        return self.keras_model

    def get_trained_model(self):
        return self.keras_model

    def show(self):
        self.keras_model.summary()

    def _exists_model(self):
        is_exist = os.path.exists(self.config["PATH"]["KERAS"])
        
        return is_exist

    def export_keras(self, target_path):
        self.keras_model = Model(
            self.keras_model.input,
            self.outputs,
            name="Nanodet_plus_m_1.5x_416",
        )
        self.keras_model.save(target_path)


class empty_class:
    pass

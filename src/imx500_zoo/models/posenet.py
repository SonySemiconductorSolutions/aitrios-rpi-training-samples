import os
import copy

from imx500_zoo.utilities.misc import EmptyClass
import imx500_zoo.utilities.posenet.config as C
from imx500_zoo.utilities.posenet.base import init_modules_base

import tensorflow as tf
from imx500_zoo.utilities.posenet.utils.model import get_personlab_model
from third_party.keraspersonlab.loss import (
    kp_map_loss_fn,
    short_offset_loss_fn,
    mid_offset_loss_fn,
)
from third_party.tensorflow.mobilenet import get_mobilenetv1_base
from imx500_zoo.utilities.posenet.utils.metrics import identity_metric

from imx500_zoo.trainers.posenet_trainer import final_model_path


def init_modules(config_i=None, fconfig=""):
    init_modules_base(config_i, fconfig)


def init_config(fconf, ini_config=None):
    config = C.Config(fconf, ini_config)
    config.parse_yaml()

    return config


class Posenet:
    """
    self.config                      : ini                 for imx500_zoo
    self.config.posenet._ini_config  : ini  original       for model
    self.config.posenet._json_config : json original       for model
    self.config.posenet              : set_config()        for train/valid
    self.config.posenet._yaml_config : yaml template+parse for quant
    self.config.posenet.model_config : init_yamlloaded()   for quant
    """

    def __init__(self, config):
        self.config = config
        self._ini_config = copy.deepcopy(config)
        self.keras_model = None

        if not hasattr(config, "posenet"):
            config.posenet = EmptyClass()

        self._init_config()
        init_modules(config_i=self.config.posenet)

    def _init_config(self):
        fjson = self.config["TRAINER"]["CONFIG"]
        self.config.posenet = init_config(fjson, self._ini_config)

    def setup(self):
        config = self.config.posenet
        print(f"height {config.IN_HEIGHT}, width {config.IN_WIDTH}")
        if config.RETRAIN_FLAG:
            custom_objects = {
                "kp_map_loss_fn": kp_map_loss_fn,
                "short_offset_loss_fn": short_offset_loss_fn,
                "mid_offset_loss_fn": mid_offset_loss_fn,
                "identity_metric": identity_metric,
            }
            self.keras_model = tf.keras.models.load_model(
                config.RETRAIN_MODEL_PATH,
                custom_objects=custom_objects,
                compile=False,
            )
        else:
            self.keras_model = get_personlab_model(get_mobilenetv1_base)

    def get(self):
        return self.keras_model

    def get_trained_model(self):
        return self.keras_model

    def show(self):
        self.keras_model.summary()

    def _exists_model(self):
        is_exist = os.path.exists(self.config["PATH"]["KERAS"])

        return is_exist

    def export_keras(self, target_path=final_model_path):
        print(f"saved model : {target_path}")
        self.keras_model.save(target_path)

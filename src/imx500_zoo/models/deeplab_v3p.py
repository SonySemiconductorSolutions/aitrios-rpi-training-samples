import os
import sys
import copy
from imx500_zoo.utilities.deeplab_v3p.deeplab import DeepLab
from third_party.kerasdeeplabv3plus.deeplabv3 import DeepLabArchitecture
from imx500_zoo.utilities import tf_utility


class DeeplabV3p:
    def __init__(self, ini):
        yaml = ini["TRAINER"]["CONFIG"]
        if not os.path.isfile(yaml):
            raise Exception("Config file does not exists: " + yaml)

        dl = DeepLab()
        param_yaml = dl.read_yaml(yaml)
        print("Config file exists: " + yaml)
        config = dl.parse_config(param_yaml)
        config.param_file = yaml
        config.ini = copy.deepcopy(ini)
        config.ini_file = sys.argv[1] if len(sys.argv) >= 2 else "" 
        config = dl.update_ini(config, ini)

        dl.config = config
        ini.deeplab = dl

        self.ini = ini

        self.model = None

    def setup(self):
        config = self.ini.deeplab.config
        tf_utility.assign_gpu(config.gpu_id)
        self.model = self.gen_model(
            model_name=config.model_name,
            pretrained_weights=config.backbone_pretrained_weights,
            n_classes=config.n_classes,
            load_weights_en=config.load_pretrained_weights,
            multi_gpu_en=False,
            base_network=config.backbone,
        )

    def gen_model(
        self,
        model_name,
        n_classes=4,
        base_network="mobilenetv2",
        load_weights_en=False,
        pretrained_weights=None,
        multi_gpu_en=False,
    ):
        dlab = self.ini.deeplab
        model = DeepLabArchitecture().Deeplabv3(
            backbone=base_network,
            classes=n_classes,
            input_tensor=None,
            input_shape=dlab.config.image_size + (3,),
            weights=pretrained_weights,
            alpha=1,
            OS=16,
        )

        if load_weights_en:
            f_weight = f"weights/{base_network}_{model_name}.h5"
            print(f"Loading weights: {f_weight} ")
            model.load_weights(f_weight)

        if multi_gpu_en:
            from keras.utils import multi_gpu_model
            model = multi_gpu_model(model, gpus=len(tf_utility.physical_gpus()))

        return model

    def export_keras(self, f_keras):
        print(f"saved model : {f_keras}")
        self.model.save(f_keras)
        tf_utility.summary(self.model, self.ini.deeplab.config.PATH_KERAS_SUMMARY)

    def get_trained_model(self):
        return self.model

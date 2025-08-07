import numpy as np
import multiprocessing
import json
import yaml
from distutils.util import strtobool

import sys
import os
from pathlib import Path
from enum import IntEnum, auto

from imx500_zoo.utilities import conv

YAML_TEMPLATE = "../src/imx500_zoo/utilities/posenet/_template_mct_work_flow.yaml"


class KeyPoint(IntEnum):
    COCO = auto()
    ARROR = auto()
    DOLL = auto()
    OTHER = auto()

    def dict():
        return {
            "coco": KeyPoint.COCO,
            "arrow": KeyPoint.ARROR,
            "doll": KeyPoint.DOLL,
        }

    def keypoint(s):
        dic = KeyPoint.dict()

        kp = KeyPoint.OTHER
        sl = s.lower()
        for k in dic.keys():
            if k in sl:
                kp = dic[k]
                break

        return kp


class Config:
    def __init__(self, config_json_path, ini_config=None):
        self.init_vars()
        self._json_file_path = config_json_path
        self._json_config = self.read_json(config_json_path)
        self._ini_config = ini_config
        if self._json_config is not None:
            self.init_ini2json()
            self.parse_json(self._json_config)
            self.init_dirs()

    def init_vars(self):
        self.KEYPOINTS = []
        self.NUM_KP = -1
        self.KEYPOINTS_DICT = {}
        self.RIGHT_KP = []
        self.LEFT_KP = []
        self.EDGES = []
        self.NUM_EDGES = -1
        self.NORM_FACTOR = 256.0
        self.KP_RADIUS = 32
        self.PEAK_THRESH = 0.004
        self.OKS_THRESH = 0.5
        self.kpt_oks_sigmas = []
        self.maxDets = [20]
        self.NMS_THRESH = 32

        self.IN_HEIGHT = 353
        self.IN_WIDTH = 481
        self.IMAGE_SHAPE = [self.IN_HEIGHT, self.IN_WIDTH, 3]
        self.OUTPUT_STRIDE = 16
        self.LOSS_WEIGHTS = {
            "heatmap": 4,
            "seg": 2,
            "short": 1,
            "mid": 0.25,
            "long": 0.125,
        }
        self.BATCH_NORM_FROZEN = True
        self.BATCH_SIZE = 10
        self.NUM_EPOCHS = 50
        self.MULTIPROCESSING_FLAG = True
        self.RETRAIN_FLAG = False
        self.GPU_ID = 0
        self.WORKERS = multiprocessing.cpu_count()

        self.ANNO_FILE_TEST = ""
        self.ANNO_FILE_VAL = ""
        self.ANNO_FILE_TRAIN = ""
        self.IMG_DIR_TEST = ""
        self.IMG_DIR_VAL = ""
        self.IMG_DIR_TRAIN = ""
        self.IMG_DIR_TRAIN_VAL = ""
        self.FINAL_MODEL_NAME = ""
        self.LOGS_PATH = ""
        self.RETRAIN_MODEL_PATH = ""
        self.EVALUATE_MODEL_PATH = ""
        self.SAVE_MODEL_PATH = ""
        self.SAVE_PREDICTIONS = ""
        self.TRAIN_MODEL_H5 = ""

        self.CONF_MAP_KEY = "Top"


    def parse_json(self, config):
        ds = config.get("DATASET", {})
        self.KEYPOINTS = ds.get("KEYPOINTS")
        self.NUM_KP = ds.get("NUM_KP", len(self.KEYPOINTS))  # Number of keypoints
        self.KEYPOINTS_DICT = ds.get("KEYPOINTS_DICT")
        self.RIGHT_KP = ds.get("RIGHT_KP")  # Indices of right and left keypoints (for flipping in augmentation)
        self.LEFT_KP = ds.get("LEFT_KP")
        self.EDGES = self.list2tuple(ds.get("EDGES", self.EDGES))  # List of edges as tuples of indices into the KEYPOINTS array (Each edge will be used twice in the mid-range offsets; once in each direction)
        self.NUM_EDGES = len(self.EDGES)
        self.NORM_FACTOR = ds.get("NORM_FACTOR")  # Normalization factor
        self.KP_RADIUS = ds.get("KP_RADIUS")  # Radius of the discs around the keypoints. Used for computing the ground truth and computing the losses. (Recommended to be a multiple of the output stride.)
        self.PEAK_THRESH = ds.get("PEAK_THRESH")  # The threshold for extracting keypoints from hough maps.
        self.OKS_THRESH = ds.get("OKS_THRESH")  # OSK metric threshold
        self.kpt_oks_sigmas = np.array(ds.get("kpt_oks_sigmas", self.kpt_oks_sigmas))  # OKS sigma for keypoints
        self.maxDets = ds.get("maxDets")  # Maximum detections
        self.NMS_THRESH = ds.get("NMS_THRESH")  # Pixel distance threshold for whether to begin a new skeleton instance (If another skeleton already has this keypoint within the threshold, it is discarded.)

        tn = config.get("TRAINING", {})
        self.IMAGE_SHAPE = tuple(tn.get("IMAGE_SHAPE", self.IMAGE_SHAPE))  # Input shape for training images (By convention s*n+1 for some integer n and s=output_stride)
        self.OUTPUT_STRIDE = tn.get("OUTPUT_STRIDE")  # Output stride of the base network (resnet101 or resnet152 in the paper) [Any convolutional stride in the original network which would reduce the output stride further is replaced with a corresponding dilation rate.]
        self.LOSS_WEIGHTS = tn.get("LOSS_WEIGHTS")  # Weights for the losses applied to the keypoint maps ('heatmap'), the binary segmentation map ('seg'), and the short-, mid-, and long-range offsets.
        self.BATCH_NORM_FROZEN = self.str2bool(tn.get("BATCH_NORM_FROZEN", "True"))  # Whether to keep the batchnorm weights frozen.
        self.BATCH_SIZE = tn.get("BATCH_SIZE", 10)  # batch size
        self.NUM_EPOCHS = tn.get("NUM_EPOCHS", 50)
        self.MULTIPROCESSING_FLAG = self.str2bool(tn.get("MULTIPROCESSING_FLAG", "True"))  # Multiprocessing flag
        self.RETRAIN_FLAG = self.str2bool(tn.get("RETRAIN_FLAG", "False"))
        self.LEARNING_RATE = tn.get("LEARNING_RATE", 0.001)
        self.LR_DECAY_STEPS = tn.get("lr_decay_steps", 1000)
        self.LR_DECAY_RATE = tn.get("lr_decay_rate", 1.00)
        self.EARLYSTOP_PATIENCE = tn.get("earlystop_patience", 25)
        self.GPU_ID = tn.get("GPU_ID", 0)
        self.WORKERS = tn.get("WORKERS")

        pt = config.get("PATH", {})
        self.ANNO_FILE_TRAIN = pt.get("ANNO_FILE_TRAIN")  # Filepath for annotation file and image directory/
        self.IMG_DIR_TRAIN = pt.get("IMG_DIR_TRAIN")
        self.ANNO_FILE_VAL = pt.get("ANNO_FILE_VAL")
        self.IMG_DIR_VAL = pt.get("IMG_DIR_VAL")
        self.ANNO_FILE_TEST = pt.get("ANNO_FILE_TEST")
        self.IMG_DIR_TEST = pt.get("IMG_DIR_TEST")
        self.IMG_DIR_TRAIN_VAL = pt.get("IMG_DIR_TRAIN_VAL")
        self.SAVE_MODEL_PATH = pt.get("SAVE_MODEL_PATH")  # Where to save the model.
        self.LOGS_PATH = pt.get("LOGS_PATH")
        self.FINAL_MODEL_NAME = pt.get("FINAL_MODEL_NAME")
        self.RETRAIN_MODEL_PATH = pt.get("RETRAIN_MODEL_PATH")  # If retrain flag set- retrain model path
        self.EVALUATE_MODEL_PATH = pt.get("EVALUATE_MODEL_PATH")  # Path to saved model for evaluation
        self.SAVE_PREDICTIONS = pt.get("SAVE_PREDICTIONS")  # Path to save model predictions

        self.TRAIN_MODEL_H5 = os.path.splitext(self.EVALUATE_MODEL_PATH)[0] + ".h5"

        vl = config.get("VISUALISE", {})
        self.CONF_MAP_KEY = vl.get("CONF_MAP_KEY")

    def init_ini2json(self):
        ini = self._ini_config
        json = self._json_config
        if ini is None:
            return

        ini_model = ini["MODEL"]
        ini_train = ini["TRAINER"]
        json_train = json["TRAINING"]
        json["DATASET"]["NUM_KP"] = int(ini_model["NUM_CLASSES"])
        isize = conv.list(ini_model["INPUT_SIZE"])
        json_train["IMAGE_SHAPE"] = [isize[0], isize[1], 3]
        self.IN_HEIGHT = isize[0]
        self.IN_WIDTH = isize[1]
        json_train["RETRAIN_FLAG"] = ini_model.get("PRE_TRAIN", "False").title()
        json_train["BATCH_SIZE"] = int(ini_train["BATCH_SIZE"])
        json_train["NUM_EPOCHS"] = int(ini_train["NUM_EPOCHS"])
        json_train["LEARNING_RATE"] = float(ini_train["LEARNING_RATE"])

        ipath = ini["PATH"]
        jpath = json["PATH"]
        model_name = ini["SOLUTION"]["NAME"]
        log = ipath["LOG"]
        log_path = os.path.join(log, model_name)
        jpath["SAVE_MODEL_PATH"] = ipath["MODEL"]
        jpath["LOGS_PATH"] = log_path
        jpath["FINAL_MODEL_NAME"] = os.path.basename(ipath["KERAS"])
        jpath["EVALUATE_MODEL_PATH"] = ipath["KERAS"]
        jpath["QUANT_MODEL_PATH"] = ipath["QUANTIZED_KERAS"]
        jpath["SAVE_PREDICTIONS"] = os.path.join(log_path, "visualize")

    def read_json(self, fjson):
        data = None
        if os.path.isfile(fjson):
            with open(fjson, "r") as fo:
                data = json.load(fo)
        else:
            print(f"Error : No File {fjson}")

        return data

    def read_yaml(self, fyaml):
        data = None
        if os.path.isfile(fyaml):
            with open(fyaml) as fo:
                data = yaml.load(fo, Loader=yaml.FullLoader)
        else:
            print(f"Error : No File {fyaml}")

        return data

    def parse_yaml(self, fyaml=YAML_TEMPLATE):
        self._yaml_file_path = fyaml
        self._yaml_config = self.read_yaml(fyaml)

        if self._yaml_config is not None:
            k = self._yaml_config["MCT_WORK_FLOW"]["KEYPOINTS"]["MODEL1"]

            d = k["DATASET"]
            d["TEST_DATA_FOLDER"] = self.IMG_DIR_TEST
            d["ANNO_FILE"] = self.ANNO_FILE_TEST
            d["REPRESENTATIVE_DATA_PATH"] = self.IMG_DIR_VAL

            m = k["MCT_WORKFLOW"]
            m["MODEL_PATH"] = self.train_model_path()
            m["INPUT_SIZE"] = f"{self.IMAGE_SHAPE[0]}x{self.IMAGE_SHAPE[1]}"
            m["NB_CLASSES"] = f"{self.NUM_KP}"

    def train_model_path(self):
        return os.path.join(self.SAVE_MODEL_PATH, self.FINAL_MODEL_NAME)

    def quantized_model_path(self):
        pt = self._json_config["PATH"]
        return pt.get("QUANT_MODEL_PATH", "")

    def eval_model_path(self):
        return self.EVALUATE_MODEL_PATH

    def init_dirs(self):
        trained = os.path.join(self.SAVE_MODEL_PATH, self.FINAL_MODEL_NAME)
        dirs = [
            self.LOGS_PATH,
            "data",
            self.SAVE_MODEL_PATH,
            os.path.dirname(trained),  # after training
            os.path.dirname(self.EVALUATE_MODEL_PATH),
            os.path.dirname(self.eval_model_path()),  # after quantization
            self.SAVE_PREDICTIONS,  # visualize
        ]

        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def is_remove_relu6(self):
        return True

    def str2bool(self, strings):
        return bool(strtobool(strings))

    def list2tuple(self, list_data):
        tuple_data = []
        for data in list_data:
            tuple_data.append(tuple(data))
        return tuple_data

    def get_permute(self):
        if self.keypoint() == KeyPoint.COCO:
            permute = [
                0,
                6,
                8,
                10,
                5,
                7,
                9,
                12,
                14,
                16,
                11,
                13,
                15,
                2,
                1,
                4,
                3,
            ]
        else:
            permute = [x for x in range(self.NUM_KP)]

        return permute

    def get_torso_dia(self, g, input_shape):
        t = self.keypoint()
        PCK_THRESH = self.OKS_THRESH
        if t == KeyPoint.ARROR:
            if g[1][2] > 0 and g[6][2] > 0:
                torso_dia = 0.2 * (
                    ((g[2][0] - g[6][0]) ** 2 + (g[2][1] - g[6][1]) ** 2) ** 0.5
                )
            else:
                torso_dia = (
                    PCK_THRESH
                    * 6.4
                    * (float((input_shape[0] + input_shape[1]) / 2) / 256.0)
                )
        elif t == KeyPoint.DOLL:
            if g[5][2] > 0 and g[8][2] > 0:
                torso_dia = 0.2 * (
                    ((g[5][0] - g[8][0]) ** 2 + (g[5][1] - g[8][1]) ** 2) ** 0.5
                )
            else:  # if either left shoulder/right hip not visible
                torso_dia = (
                    PCK_THRESH
                    * 6.4
                    * (float((input_shape[0] + input_shape[1]) / 2) / 256.0)
                )
        else:  # COCO or OTHER
            if g[4][2] > 0 and g[7][2] > 0:
                torso_dia = 0.2 * (
                    ((g[4][0] - g[7][0]) ** 2 + (g[4][1] - g[7][1]) ** 2) ** 0.5
                )
            else:  # if either left shoulder/right hip not visible
                torso_dia = (
                    PCK_THRESH
                    * 6.4
                    * (float((input_shape[0] + input_shape[1]) / 2) / 256.0)
                )

        return torso_dia

    def keypoint(self):
        key = "TYPE"
        dataset = self._json_config["DATASET"]
        if key in dataset:
            s = dataset[key]
        else:
            s = os.path.basename(self._json_file_path)

        return KeyPoint.keypoint(s)

    def get_classnames(self):
        return self._json_config["DATASET"]["KEYPOINTS_DICT"]

    def init_yamlloaded(self, model_config):
        self.model_config = model_config
        self.root_dir = self.model_config["root_folder"]
        self.__initialise_config_params()

    def __initialise_config_params(self):
        """Initializing the essential configuration parameters required for the framework."""

        # create folder if not exists
        dweights = (
            "local_trained_weights"
            if self.model_config["is_local_weights"]
            else "sts_trained_weights"
        )
        self.generated_root_folder = os.path.join(
            self.root_dir,
            "generated",
            self.model_config["task_name"],
            f"{self.model_config['model_name']}_{self.model_config['dataset_name']}",
            dweights,
        )

        Path(self.generated_root_folder).mkdir(parents=True, exist_ok=True)
        ddataset = (
            f"{self.model_config['model_name']}_{self.model_config['dataset_name']}"
        )
        self.mct_workflow_float32_float32_tflite_path = os.path.join(
            self.generated_root_folder,
            f"{ddataset}_float32.tflite",
        )
        self.mct_workflow_mct_float32_tflite_mct_float32_tflite_path = os.path.join(
            self.generated_root_folder,
            f"{ddataset}_mct_float32.tflite",
        )
        self.mct_workflow_mct_float32_tflite_mct_h5_path = os.path.join(
            self.generated_root_folder,
            f"{ddataset}_mct_quant_h5.h5",
        )
        self.mct_workflow_mct_int8_cpu_int8_mct_tflite_path = os.path.join(
            self.generated_root_folder,
            f"{ddataset}_mct_int8.tflite",
        )

        self.evb_model_path = os.path.join(self.generated_root_folder, "evb_model")

        # create folder if not exists for input tensors
        self.input_tensors_folder = os.path.join(
            self.generated_root_folder, "input_tensors"
        )
        Path(self.input_tensors_folder).mkdir(parents=True, exist_ok=True)
        self.input_scale = None
        self.input_format = None

        # Paths for generated files
        self.packer_output_dir = os.path.join(
            self.generated_root_folder, "custom_networks"
        )
        self.custom_network_dir = os.path.join(
            self.generated_root_folder, "custom_networks", "CustomNet_Test_pv2"
        )
        self.input_tensor_path = os.path.join(
            self.generated_root_folder, "input_tensors"
        )
        self.output_tensor_path = os.path.join(
            self.generated_root_folder, "output_tensors"
        )
        self.config_yaml_path = os.path.join(
            self.generated_root_folder, "custom_networks", "config.yaml"
        )
        self.dnnparams_xml_path = os.path.join(
            self.generated_root_folder, "custom_networks", "dnnParams.xml"
        )
        self.input_tensor_list_file_path = os.path.join(
            self.generated_root_folder,
            "input_tensors",
            f"{self.model_config['model_name']}_{self.model_config['dataset_name']}.list",
        )
        Path(self.packer_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.custom_network_dir).mkdir(parents=True, exist_ok=True)
        Path(self.input_tensor_path).mkdir(parents=True, exist_ok=True)
        Path(self.output_tensor_path).mkdir(parents=True, exist_ok=True)


if __name__ == "config":
    args = sys.argv
    json_file_path = args[1]
    print(json_file_path)
    config = Config(json_file_path)

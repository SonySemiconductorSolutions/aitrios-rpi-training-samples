import os
import yaml
import sys
import datetime
import shutil

# use cpu if gpu not available
import tensorflow as tf
from imx500_zoo.utilities.posenet.quant.common.tflite_converter import (
    TFLiteConverter,
)
from imx500_zoo.utilities.posenet.quant.common.tflite_Inference import (
    TFliteInference,
)
from imx500_zoo.utilities.posenet.config import Config
import warnings

warnings.filterwarnings("ignore")

logger = None

config = None


def set_config(config_i):
    global config
    global logger

    config = config_i
    logger = config.quant_logger


class MctTflite:
    def __init__(self, config):
        self.config = config

    def quantize(self, model, dataloader_quant):
        quant = MCTWorkFlow()
        quant.model = model
        quant.dataloader_quant = dataloader_quant
        quant._config_data = self.config.posenet._yaml_config
        quant.config_zoo = self.config
        quant.execute()


class YamlConfigParser:
    def __init__(self, config_file):
        self._config_file = config_file
        self._config_data = self._read_yaml()

    def _read_yaml(self):
        with open(self._config_file) as file_handle:
            self._config_data = yaml.load(file_handle, Loader=yaml.FullLoader)
        return self._config_data


class MCTWorkFlow(YamlConfigParser):
    def __init__(self, config_file=""):
        if os.path.isfile(config_file):
            super().__init__(os.path.abspath(config_file))
        self.dataloader_quant = None

    def _get_classnames(self):
        return config.get_classnames()

    def execute(self):
        root_folder = self._config_data["ROOT_FOLDER"]
        sdsp_tool_chain_path = self._config_data["SDSP_TOOL_CHAIN_PATH"]
        for task_name, model_object in self._config_data["MCT_WORK_FLOW"].items():
            for model_object_key, model_object_value in model_object.items():
                self.model_config = {}
                self.model_config["root_folder"] = root_folder
                self.model_config["sdsp_tool_chain_path"] = sdsp_tool_chain_path
                self.model_config["task_name"] = task_name

                if task_name == "KEYPOINTS":
                    model_name = model_object_value["NAME"]
                    is_local_weights = model_object_value["IS_LOCAL_WEIGHTS"]
                    dataset_name = model_object_value["DATASET"]["NAME"]
                    dataset_test_data_folder = model_object_value["DATASET"][
                        "TEST_DATA_FOLDER"
                    ]
                    annotation_file = model_object_value["DATASET"]["ANNO_FILE"]
                    dataset_representative_data_path = model_object_value["DATASET"][
                        "REPRESENTATIVE_DATA_PATH"
                    ]

                    mct_workflow_model_path = model_object_value["MCT_WORKFLOW"][
                        "MODEL_PATH"
                    ]

                    mct_workflow_mct_int8_imx500_enable = model_object_value[
                        "MCT_WORKFLOW"
                    ]["MCT_INT8_IMX500"]["ENABLE"]

                    # 'input size' and 'no of classes'
                    input_size_str = model_object_value["MCT_WORKFLOW"]["INPUT_SIZE"]
                    nb_classes_str = model_object_value["MCT_WORKFLOW"]["NB_CLASSES"]
                    input_size_list = input_size_str.split("x")

                    input_size = (
                        int(input_size_list[0]),
                        int(input_size_list[1]),
                    )
                    self.input_size = input_size
                    nb_kp = int(nb_classes_str)
                    self.nb_kp = nb_kp

                    dataset_test_data_folder = os.path.join(
                        root_folder, dataset_test_data_folder
                    )
                    dataset_representative_data_path = os.path.join(
                        root_folder, dataset_representative_data_path
                    )

                    mct_workflow_model_path = os.path.join(
                        root_folder, mct_workflow_model_path
                    )

                    logger.info(model_name)
                    logger.info(is_local_weights)
                    logger.info(dataset_name)
                    logger.info(dataset_test_data_folder)
                    logger.info(dataset_representative_data_path)
                    logger.info(config._json_file_path)
                    print("Keypoints H5 model path : ", mct_workflow_model_path)
                    logger.info(mct_workflow_mct_int8_imx500_enable)

                    self.model_config["model_name"] = model_name
                    self.model_config["is_local_weights"] = is_local_weights
                    self.model_config["dataset_name"] = dataset_name
                    self.model_config["dataset_test_data_folder"] = (
                        dataset_test_data_folder
                    )
                    self.model_config["dataset_representative_data_path"] = (
                        dataset_representative_data_path
                    )
                    self.model_config["mct_workflow_model_path"] = (
                        mct_workflow_model_path
                    )
                    self.model_config["mct_workflow_mct_int8_imx500_enable"] = (
                        mct_workflow_mct_int8_imx500_enable
                    )
                    self.model_config["annotation_file"] = annotation_file
                    self.model_config["nb_classes"] = nb_kp
                    self.model_config["input_size"] = input_size

                    c = Config("")
                    c.init_yamlloaded(self.model_config)
                    c.config_json = config
                    config.model_config = c.model_config
                    self.config = c
                    tflite_converter = TFLiteConverter(self.config, input_size)
                    tflite_inference = TFliteInference(self.config)

                    # getting h5 model name
                    input_h5_model_path = mct_workflow_model_path
                    print("input_h5_model_path :", input_h5_model_path)
                    model_path_splitted = input_h5_model_path.split("/")
                    input_model_name_list = model_path_splitted[-1:][0].split(".")[:-1]
                    input_model_name = input_model_name_list[0]
                    if len(input_model_name_list) > 1:
                        for idx in range(1, len(input_model_name_list)):
                            input_model_name = (
                                input_model_name + "_" + input_model_name_list[idx]
                            )
                    print("input_model_name :", input_model_name)

                    self.generated_output_root_folder_path = os.path.join(
                        self.config.root_dir,
                        "generated",
                        self.config.model_config["task_name"],
                        f"{self.config.model_config['model_name']}_{self.config.model_config['dataset_name']}",
                    )

                    # if MCT_INT8_IMX500 is TRUE in mct_work_flow.yaml
                    if self.config.model_config["mct_workflow_mct_int8_imx500_enable"]:
                        mct_keras_model_path = config.quantized_model_path()

                        path, ext = os.path.splitext(mct_keras_model_path)
                        mct_int8_cpu_model_path = f"{path}.{MCTWorkFlow.tflite()}"

                        tflite_converter.keypoint_mct_convert(
                            input_h5_model_path,
                            mct_int8_cpu_model_path,
                            mct_keras_model_path,
                        )
                        print(f"write quantized output: {mct_int8_cpu_model_path}")

                        logs = tflite_inference.keypoint_inference(
                            mct_int8_cpu_model_path,
                            dataset_test_data_folder,
                            annotation_file,
                            input_size,
                            nb_kp,
                            self.dataloader_quant,
                        )
                        logs["MODEL"] = mct_int8_cpu_model_path
                        self.config_zoo.results.quant = logs

    def tflite():
        return "tflite"


def backup_file(ffrom):
    if os.path.isfile(ffrom):
        ftail = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fto = f"{os.path.join(os.path.dirname(ffrom), os.path.splitext(os.path.basename(ffrom))[0])}__{ftail}.h5"
        print(f"warning : file is moved from {ffrom} to {fto}")
        shutil.move(ffrom, fto)


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
        except Exception as e:
            print(f"GPU unavailable : {e}")
    args = sys.argv
    config_yaml_file = args[1]
    mct_work_flow = MCTWorkFlow(config_yaml_file)
    mct_work_flow.execute()

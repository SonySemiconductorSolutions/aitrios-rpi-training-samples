import json
from PIL import Image
import numpy as np
import cv2
from imx500_zoo import utilities
from third_party.mct.coco_evaluation import DatasetGenerator


DOWNLOAD_ANNOTATION = (
    r"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
DOWNLOAD_DATASET = r"http://images.cocodataset.org/zips/val2017.zip"


class CocoVal2017:
    def __init__(self, config):
        self.config = config
        self.coco_label_map = self.set_coco_label_map()

    def download_annotations(self, data_path):
        utilities.download_zip(
            DOWNLOAD_ANNOTATION,
            data_path,
            "annotations",
        )

    def download_dataset(self, data_path):
        utilities.download_zip(
            DOWNLOAD_DATASET,
            data_path,
            "val2017",
        )

    def setup(self):
        self.input_size = int(self.config["MODEL"]["INPUT_SIZE"])
        self.img_mean = json.loads(self.config["MODEL"]["MEAN"])
        self.img_std = json.loads(self.config["MODEL"]["STD"])
        self.batch_size = int(self.config["QUANTIZER"]["BATCH_SIZE"])

        self.data_path = self.config["PATH"]["DATA"]
        self.download_annotations(self.data_path)
        self.download_dataset(self.data_path)

        dataset_folder = self.data_path + "/val2017"
        annotation_file_path = (
            self.data_path + "/annotations/instances_val2017.json"
        )
        self.dataloader_quant, self.dataloader_eval = self.load_dataset(
            dataset_folder, annotation_file_path
        )

        self.set_target_pathes(dataset_folder, annotation_file_path)

    def set_target_pathes(self, dataset_folder, annotation_file_path):
        self.config["PATH"]["DATASET"] = dataset_folder
        self.config["PATH"]["ANNOTATION_FILE"] = annotation_file_path

    def preprocess(self, img):
        input_size = self.input_size
        if Image.isImageType(img):
            img = img.resize((input_size, input_size))
        elif type(img) is np.ndarray:
            img = cv2.resize(img, (input_size, input_size))
        else:
            raise Exception(
                "TypeError: Only PIL.Image.Image and numpy.ndarray are supported"
            )
        img_mean = np.array(self.img_mean)
        img_std = np.array(self.img_std)
        preprocessed_img = (img - img_mean) / img_std
        return preprocessed_img

    def load_dataset(self, dataset_folder, annotation_file_path):
        quantize_data_gen = DatasetGenerator(
            dataset_folder=dataset_folder,
            annotation_file=annotation_file_path,
            preprocess=self.preprocess,
            batch_size=self.batch_size,
        )

        evaluate_data_gen = DatasetGenerator(
            dataset_folder=dataset_folder,
            annotation_file=annotation_file_path,
            preprocess=self.preprocess,
            batch_size=self.batch_size,
        )

        return quantize_data_gen, evaluate_data_gen

    def get_loaders(self):
        return None, None, self.dataloader_quant, self.dataloader_eval

    def get_single_data(self):
        print("validator_size:", self.validloader.shape)
        print("validator[index]_size:", self.validloader[self.index].shape)
        single_data = self.validloader[self.index]
        single_data = single_data[np.newaxis, ...]
        return single_data

    def set_coco_label_map(self):
        coco_label_map = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        }
        return coco_label_map

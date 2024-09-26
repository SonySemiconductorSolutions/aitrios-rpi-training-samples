import torch
import os

import imx500_zoo.datasets
from imx500_zoo import utilities
import third_party.nanodet.card_detection as NC

import albumentations as albu
import albumentations.pytorch
import json
import numpy as np
import cv2

from third_party.mct.coco_evaluation import DatasetGenerator

from PIL import Image


def np_to_pil(img):
    return Image.fromarray(img)


DOWNLOAD_DATASET = r"https://github.com/SonySemiconductorSolutions/aitrios-rpi-dataset-sample/raw/refs/heads/main/card_detection.zip"


class CardDetection:
    def __init__(self, config):
        # call from imx500_zoo.py/Solution.setup_data(), setup_model() is executed
        if config.is_retrain:
            config.is_train_nanodet = (
                config["TRAINER"]["NAME"].lower().__contains__("nanodet")
            )
            if not hasattr(config, "nanodet"):
                config.nanodet = empty_class()
        else:
            config.is_train_nanodet = False

        self.config = config

        self.c = NC.CardDetection(config)

    def _download(self, data_path):
        utilities.download_zip(
            DOWNLOAD_DATASET,
            data_path,
            "annotations",
            target_name="card_detection.zip",
        )

    def _read_annotations(self, annotations_path):
        json_open = open(annotations_path, "r")
        json_load = json.load(json_open)

        data = {"file_path": [], "bbox": [], "category_id": []}

        # get image path
        for image in json_load["images"]:
            data["file_path"].append(image["file_name"])

        # get bbox and category_id
        for annotations in json_load["annotations"]:
            img_bbox = []
            for i in range(4):
                img_bbox.append(int(annotations["bbox"][i]))

            data["bbox"].append(img_bbox)
            data["category_id"].append(annotations["category_id"])

        return data

    def setup(self):
        # call from imx500_zoo.py/Solution.setup_data(), setup_model() is executed
        self.input_size = int(self.config["MODEL"]["INPUT_SIZE"])
        self.img_mean = json.loads(self.config["MODEL"]["MEAN"])
        self.img_std = json.loads(self.config["MODEL"]["STD"])

        self.quant_batch_size = int(self.config["QUANTIZER"]["BATCH_SIZE"])

        self.train_batch_size = int(self.config["TRAINER"]["BATCH_SIZE"])
        self.train_num_workers = int(self.config["TRAINER"]["NUM_WORKERS"])

        self.data_path = "./data/card_detection"
        self.train_path = os.path.join(self.data_path, "train")
        self.valid_path = os.path.join(self.data_path, "val")
        self.annotations_path = os.path.join(self.data_path, "annotations")

        annotation_val_file_path = os.path.join(
            self.annotations_path, "annotations_val.json"
        )
        self._set_target_pathes(self.valid_path, annotation_val_file_path)

        self._download(self.data_path)

        if self.config.is_train_nanodet:
            self._setup_retrain()
        else:
            self._setup_with_pth()

        self.dataloader_quant, self.dataloader_eval = self._load_dataset(
            self.valid_path, annotation_val_file_path
        )

    def _setup_retrain(self):
        c = self.c

        c.input_size = self.input_size
        c.img_mean = self.img_mean
        c.img_std = self.img_std
        c.batch_size = self.train_batch_size
        c.num_workers = self.train_num_workers
        c.data_path = self.data_path
        c.train_path = self.train_path
        c.valid_path = self.valid_path
        c.annotations_path = self.annotations_path

        c.setup_retrain()

        self.trainloader = c.trainloader
        self.validloader = c.validloader

    def _setup_with_pth(self):
        input_width_size = int(self.config["MODEL"]["INPUT_SIZE"])
        input_height_size = int(input_width_size * 0.75)

        train_data = self._read_annotations(
            "./data/card_detection/annotations/annotations_train.json"
        )
        valid_data = self._read_annotations(
            "./data/card_detection/annotations/annotations_val.json"
        )

        self.transform = {
            "train": albu.Compose(
                [
                    albu.Resize(input_height_size, input_width_size),
                    albu.HorizontalFlip(p=0.5),
                    albu.VerticalFlip(p=0.5),
                    albu.Rotate(limit=(-60, 60), p=0.5),
                    albu.ShiftScaleRotate(scale_limit=(0, 0), p=0.5),
                    albu.HueSaturationValue(p=0.5),
                    albu.ElasticTransform(p=0.5),
                    albu.Normalize(),
                    albu.pytorch.ToTensorV2(),
                ],
                bbox_params=albu.BboxParams(
                    format="coco", label_fields=["class_labels"]
                ),
            ),
            "valid": albu.Compose(
                [
                    albu.Resize(input_height_size, input_width_size),
                    albu.Normalize(),
                    albu.pytorch.ToTensorV2(),
                ],
                bbox_params=albu.BboxParams(
                    format="coco", label_fields=["class_labels"]
                ),
            ),
        }

        self.dataset_train = getattr(imx500_zoo.datasets, "AlbuImageFolder")(
            train_data["file_path"],
            train_data["bbox"],
            train_data["category_id"],
            self.transform["train"],
        )
        self.dataset_valid = getattr(imx500_zoo.datasets, "AlbuImageFolder")(
            valid_data["file_path"],
            valid_data["bbox"],
            valid_data["category_id"],
            self.transform["valid"],
        )

        self.trainloader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=self.train_num_workers,
        )

        self.validloader = torch.utils.data.DataLoader(
            self.dataset_valid,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=self.train_num_workers,
        )

    def _set_target_pathes(self, dataset_folder, annotation_file_path):
        self.config["PATH"]["DATASET"] = dataset_folder
        self.config["PATH"]["ANNOTATION_FILE"] = annotation_file_path

    def _load_dataset(self, dataset_folder, annotation_file_path):
        quantize_data_gen = DatasetGenerator(
            dataset_folder=dataset_folder,
            annotation_file=annotation_file_path,
            preprocess=self._preprocess,
            batch_size=self.quant_batch_size,
        )

        evaluate_data_gen = DatasetGenerator(
            dataset_folder=dataset_folder,
            annotation_file=annotation_file_path,
            preprocess=self._preprocess,
            batch_size=self.quant_batch_size,
        )

        return quantize_data_gen, evaluate_data_gen

    def _preprocess(self, img):
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

    def get_loaders(self):
        return (
            self.trainloader,
            self.validloader,
            self.dataloader_quant,
            self.dataloader_eval,
        )

    def get_single_data(self, index):
        return self.dataset_valid[index]


class empty_class:
    pass

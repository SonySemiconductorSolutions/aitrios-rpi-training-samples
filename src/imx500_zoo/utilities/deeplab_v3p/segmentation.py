import cv2
import numpy as np
import random
import os

import albumentations as A
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomBrightnessContrast,
    GaussianBlur,
    Resize,
    RandomCrop,
    CLAHE,
)
from albumentations.augmentations.geometric.transforms import Affine

from keras.utils import Sequence
from sklearn.utils import class_weight

BASENAME_LIST = "label_data.txt"
IMAGE_DIR = "JPEGimages"
IMAGE_EXT = ".jpg"
LABEL_DIR = "SegmentationClassRaw"
LABEL_EXT = ".png"


class Segmentation(Sequence):
    def __init__(
        self,
        dir="./data_utils/Datasets",
        mode="train",
        config=None,
        n_classes=21,
        batch_size=1,
        seed=7,

        blur=0,
        resize_shape=(120, 120),
        crop_shape=None,
        h_flip_en=True,
        v_flip_en=False,
        brightness=0.1,
        rotation=5.0,
        zoom=0.1,
        contrast_en=True,
        probability=0.5,

        scale=1.0 / 256.0,
    ):
        self.config = config
        self.n_classes = n_classes
        self.n_classes_wo_void = n_classes - 1
        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.scale = scale
        self.is_memory_allocated = False

        if seed:
            random.seed = seed
        self.setup_list(dir, mode)

        self.pipeline = self.get_pipeline(
            blur=blur,
            resize_shape=resize_shape,
            crop_shape=crop_shape,
            h_flip_en=h_flip_en,
            v_flip_en=v_flip_en,
            brightness=brightness,
            rotation=rotation,
            zoom=zoom,
            contrast_en=contrast_en,
            pr=probability,
        )

        self.probability = probability

    def __len__(self):
        return len(self.f_images) // self.batch_size

    def __getitem__(self, i):
        """
        batch_size of annoteted data
        """
        self.setup_memory()
        i_from = i * self.batch_size
        i_to = i_from + self.batch_size
        files = zip(
            self.f_images[i_from:i_to],
            self.f_labels[i_from:i_to],
        )
        for n, (f_image, f_label) in enumerate(files):
            image, label = self.augmentation(f_image, f_label)
            self.get_label(image, label, n, f_label)

        return self.get_item()

    def setup_list(self, folder, mode):
        """
        generate lists of images and labels with random order
        """
        d_root = os.path.join(folder, mode)
        f_names = os.path.join(d_root, BASENAME_LIST)
        d_image = os.path.join(d_root, IMAGE_DIR)
        d_label = os.path.join(d_root, LABEL_DIR)

        with open(f_names) as file:
            names = [line.strip() for line in file if line.strip()]
        random.shuffle(names)
        self.f_images = []
        self.f_labels = []
        for n in names:
            self.f_images.append(os.path.join(d_image, n + IMAGE_EXT))
            self.f_labels.append(os.path.join(d_label, n + LABEL_EXT))

        print(f"{mode} = total images: {len(names)}")

    def setup_memory(self):
        """
        allocate memories for a batch
        """
        if self.is_memory_allocated:
            return

        if self.crop_shape:
            shape = self.crop_shape
        elif self.resize_shape:
            shape = self.resize_shape
        else:
            raise Exception("No image dimensions specified!")
        height = shape[1]
        width = shape[0]
        self.height = height
        self.widht = width
        pixels = height * width
        self.IMAGES = np.zeros(
            (self.batch_size, height, width, 3),
            dtype="float32",
        )
        self.LABELS = np.zeros(
            (self.batch_size, pixels, 1),
            dtype="float32",
        )
        self.PRED = np.zeros(
            (self.batch_size, pixels),
            dtype="float32",
        )
        self.is_memory_allocated = True

    def get_label(self, image, label, n, f_label=""):
        # Process label
        label = label.astype("int32")
        y = label.flatten()
        y[y > self.n_classes_wo_void] = self.n_classes

        # Compute class weights
        valid_pixels = y[y < self.n_classes]
        u_classes = np.unique(valid_pixels)
        if len(u_classes) == 0:
            print(f"Warning : no IDs in label {f_label}")
            return

        weights = class_weight.compute_class_weight(
            "balanced", classes=u_classes, y=valid_pixels
        )
        for i in range(u_classes.size):
            self.PRED[n][y == u_classes[i]] = weights[i]
        self.PRED[n][y >= self.n_classes] = 0

        # Scale image and store
        self.IMAGES[n] = image * self.scale
        self.LABELS[n] = np.expand_dims(y, -1)

    def get_item(self):
        pred_mask = {"pred_mask": self.PRED}
        self.LABELS[self.LABELS > self.n_classes_wo_void] = 0

        return self.IMAGES, self.LABELS, pred_mask

    def augmentation(self, image_path, label_path):
        """
        tranformed set of image and label
        """
        transform = A.Compose(self.pipeline)

        image = cv2.imread(image_path)
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        augmented = transform(image=image, mask=mask)
        augmented_image = augmented["image"]
        augmented_mask = augmented["mask"]

        return augmented_image, augmented_mask

    def get_pipeline(
        self,
        blur,
        resize_shape,
        crop_shape,
        h_flip_en,
        v_flip_en,
        brightness,
        rotation,
        zoom,
        contrast_en,
        pr=0.5,
    ):
        """
        generate pipeline for augmentation
        """
        pipeline = []
        if blur:
            pipeline.append(GaussianBlur(blur_limit=(blur, blur), p=pr))
        if resize_shape and not crop_shape:
            pipeline.append(
                Resize(
                    height=resize_shape[1],
                    width=resize_shape[0],
                    interpolation=cv2.INTER_AREA,
                )
            )
        if crop_shape:
            pipeline.append(RandomCrop(height=crop_shape[1], width=crop_shape[0]))
        if h_flip_en:
            pipeline.append(HorizontalFlip(p=pr))
        if v_flip_en:
            pipeline.append(VerticalFlip(p=pr))
        if brightness:
            pipeline.append(
                RandomBrightnessContrast(
                    brightness_limit=[-brightness, brightness], contrast_limit=0, p=pr
                )
            )
        if rotation or zoom:
            pipeline.append(
                Affine(
                    scale=(1 - zoom, 1 + zoom),
                    rotate=(-rotation, rotation),
                    keep_ratio=True,
                    p=pr,
                )
            )
        if contrast_en:
            pipeline.append(CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=pr))

        return pipeline

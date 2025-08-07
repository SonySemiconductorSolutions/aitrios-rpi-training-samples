import cv2
import numpy as np
from pycocotools.coco import COCO
import os
from enum import IntEnum, auto
from .utils.augmentation import Augmentation

from third_party.keraspersonlab.data_parser import (
    get_ground_truth,
)
import tensorflow as tf
from keras.utils import Sequence
import random
import math
import re


ANNO_FILE_TRAIN = None
IMG_DIR_TRAIN = None
IMG_DIR_VAL = None
IMG_DIR_TEST = None
IMG_DIR_TRAIN_VAL = None
SSS_data_filepath = None
img_shape = None
NUM_EDGES = None

config = None


def set_config(config_i):
    global config
    global ANNO_FILE_TRAIN
    global IMG_DIR_TRAIN
    global IMG_DIR_VAL
    global IMG_DIR_TEST
    global IMG_DIR_TRAIN_VAL
    global SSS_data_filepath
    global img_shape
    global NUM_EDGES

    config = config_i

    ANNO_FILE_TRAIN = config.ANNO_FILE_TRAIN
    IMG_DIR_TRAIN = config.IMG_DIR_TRAIN
    IMG_DIR_VAL = config.IMG_DIR_VAL
    IMG_DIR_TEST = config.IMG_DIR_TEST
    IMG_DIR_TRAIN_VAL = config.IMG_DIR_TRAIN_VAL
    SSS_data_filepath = [
        IMG_DIR_TRAIN,
        IMG_DIR_VAL,
        IMG_DIR_TRAIN_VAL,
        IMG_DIR_TEST,
    ]
    img_shape = [config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]]
    NUM_EDGES = config.NUM_EDGES


class DataGenType(IntEnum):
    TRAIN = auto()
    VALID = auto()
    QUANT = auto()


class DataGenerator(Sequence):
    def __init__(
        self,
        BATCH_SIZE,
        phase="train",
        train_val_split=None,
        seed=None,
        return_kp=False,
        testDataFldr=None,  # extend for QUANT
        annotation_file=None,
        input_size=None,
        nb_kp=None,
        dg_type=DataGenType.TRAIN,
    ):
        self.dg_type = dg_type

        if phase == "train":
            self.IMG_DIR = config.IMG_DIR_TRAIN
            self.ANNO_FILE = config.ANNO_FILE_TRAIN

        elif phase == "val":
            self.IMG_DIR = config.IMG_DIR_VAL
            self.ANNO_FILE = config.ANNO_FILE_VAL

        elif phase == "trainval" or phase == "valtrain":
            assert train_val_split > 0 and train_val_split < 1
            self.IMG_DIR = config.IMG_DIR_TRAIN_VAL
            self.ANNO_FILE = config.ANNO_FILE_TRAIN_VAL

        else:
            if self.is_quant():
                self.IMG_DIR = testDataFldr
                self.ANNO_FILE = annotation_file
            else:
                self.IMG_DIR = config.IMG_DIR_TEST
                self.ANNO_FILE = config.ANNO_FILE_TEST

        self.phase = phase
        self.BATCH_SIZE = BATCH_SIZE
        self.train_val_split = train_val_split
        self.coco = COCO(self.ANNO_FILE)
        self.return_kp = return_kp

        self.valid_img_ids = list(self.coco.imgs.keys())

        self.datasetlen = len(self.valid_img_ids)
        self.id = 0
        if (train_val_split and phase == "trainval") or (
            train_val_split and phase == "valtrain"
        ):
            self.train_len = int(np.ceil(self.datasetlen * self.train_val_split))
            self.val_len = int(self.datasetlen) - self.train_len
            self.img_ids_train = self.valid_img_ids[: self.train_len]
            self.img_ids_val = self.valid_img_ids[self.train_len :]
        else:
            self.train_len = int(self.datasetlen)
            self.val_len = int(self.datasetlen)
            self.img_ids_train = self.valid_img_ids
            self.img_ids_val = self.valid_img_ids

        if self.phase == "train" or self.phase == "trainval":
            self.datasetlen = self.train_len
        elif self.phase == "val" or self.phase == "valtrain":
            self.datasetlen = self.val_len
        else:
            self.datasetlen = len(self.valid_img_ids)

        self.seed = seed

    def __len__(self):
        return math.ceil(self.datasetlen // self.BATCH_SIZE)

    def is_quant(self):
        return self.dg_type == DataGenType.QUANT

    def on_epoch_end(self):
        def seed_fun():
            if self.seed is None:
                seed = np.random.randint(1e6)
            else:
                seed = self.seed
            return seed

        # Shuffle dataset for next epoch
        random.shuffle(self.img_ids_train)
        random.shuffle(self.img_ids_val)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

    def get_valid_img_ids(self):
        self.valid_img_ids = []

        for img_id in self.img_ids:
            check_flag = []
            filepath_flag = False
            img_anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            image_filename = self.coco.imgs[img_id]["file_name"]
            clean_imagename = re.sub(r"[^a-zA-Z0-9._-]+", "_", image_filename).replace(
                "__", "_"
            )
            clean_imagename = clean_imagename.replace("-", "_")
            for anno in img_anns:
                check_flag = anno["keypoints"]
                break

            if self.phase == "train":
                SSS_data_filepath_iter = SSS_data_filepath[0:1]
            elif self.phase == "val":
                SSS_data_filepath_iter = SSS_data_filepath[1:2]
            elif self.phase == "trainval" or self.phase == "valtrain":
                SSS_data_filepath_iter = SSS_data_filepath[2:3]
            else:
                SSS_data_filepath_iter = SSS_data_filepath[3:4]

            for img_dir in SSS_data_filepath_iter:
                filepath_check = os.path.join(img_dir, clean_imagename)

                if os.path.exists(filepath_check):
                    filepath_flag = True

            if len(check_flag) > 0 and filepath_flag:
                self.valid_img_ids.append(img_id)

    def get_batch(self, i):
        h, w, c = config.IMAGE_SHAPE
        imgs_batch = np.zeros((self.BATCH_SIZE, h, w, c)).astype("float32")
        kp_maps_batch = np.zeros((self.BATCH_SIZE, h, w, config.NUM_KP)).astype(
            "float32"
        )
        short_offsets_batch = np.zeros(
            (self.BATCH_SIZE, h, w, 2 * config.NUM_KP)
        ).astype("float32")
        mid_offsets_batch = np.zeros(
            (self.BATCH_SIZE, h, w, 4 * (config.NUM_EDGES))
        ).astype("float32")
        keypoints_return_dict = {}

        if self.phase == "train" or self.phase == "trainval":
            give_id_list = self.img_ids_train[
                i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE
            ]

        elif self.phase == "val" or self.phase == "valtrain":
            give_id_list = self.img_ids_val[
                i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE
            ]

        else:
            give_id_list = self.valid_img_ids[
                i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE
            ]

        for j, img_id in enumerate(give_id_list):
            image_filename = self.coco.imgs[img_id]["file_name"]
            clean_imagename = re.sub(r"[^a-zA-Z0-9._-]+", "_", image_filename).replace(
                "__", "_"
            )
            clean_imagename = clean_imagename.replace("-", "_")
            idxret = clean_imagename if self.is_quant() else j
            keypoints_return_dict[str(idxret)] = {}
            filepath = os.path.join(self.IMG_DIR, clean_imagename)
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, c = img.shape
            # read the annotation, and get the keypoints and masks

            keypoints = []
            area_list = []
            img_anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

            for anno in img_anns:
                keypoints.append([int((x)) for x in anno["keypoints"]])
                # Scaling of area based on resizing of image
                area_list.append(
                    (anno["area"] * config.IMAGE_SHAPE[0] * config.IMAGE_SHAPE[1])
                    / (w * h)
                )

            # code added for correction in annotation for data at end points of image
            for batch in range(len(keypoints)):
                for idx0, val in enumerate(keypoints[batch]):
                    idx = idx0 + 1
                    if (idx % 3 == 1 and val == w) or (idx % 3 == 2 and val == h):
                        keypoints[batch][idx0] = keypoints[batch][idx0] - 1
            # code end

            kp = np.reshape(keypoints, (-1, config.NUM_KP, 3))

            # do augmentation only when it is training else only do resizing
            if not self.return_kp and self.phase == "train":
                img, kp = Augmentation.augment(img, kp, is_aug=True)
            else:
                img, kp = Augmentation.augment(img, kp, is_aug=False)

            # get ground truth from keypoints
            kp = [np.squeeze(k) for k in np.split(kp, kp.shape[0], axis=0)]

            keypoints_return_dict[str(idxret)]["keypoints"] = kp
            keypoints_return_dict[str(idxret)]["area"] = area_list

            kp_maps, short_offsets, mid_offsets = get_ground_truth(kp)

            self.id += 1
            # normalization for input images
            img = Augmentation.preprocess(img)

            sample = [
                img.astype("float32"),
                kp_maps.astype("float32"),
                short_offsets.astype("float32"),
                mid_offsets.astype("float32"),
            ]

            imgs_batch[j] = sample[0]
            kp_maps_batch[j] = sample[1]
            short_offsets_batch[j] = sample[2]
            mid_offsets_batch[j] = sample[3]

        if self.return_kp:
            return [
                imgs_batch,
                kp_maps_batch,
                short_offsets_batch,
                mid_offsets_batch,
                keypoints_return_dict,
            ]

        return [
            imgs_batch,
            kp_maps_batch,
            short_offsets_batch,
            mid_offsets_batch,
        ]

    def __getitem__(self, index):
        h, w, c = config.IMAGE_SHAPE

        if self.return_kp:
            (
                imgs_batch,
                kp_maps_batch,
                short_offsets_batch,
                mid_offsets_batch,
                keypoints_return_dict,
            ) = self.get_batch(index)

            return (
                imgs_batch,
                (kp_maps_batch, short_offsets_batch, mid_offsets_batch),
                keypoints_return_dict,
            )

        else:
            (
                imgs_batch,
                kp_maps_batch,
                short_offsets_batch,
                mid_offsets_batch,
            ) = self.get_batch(index)

            return imgs_batch, (
                kp_maps_batch,
                short_offsets_batch,
                mid_offsets_batch,
            )


class DataGeneratorTFDS(Sequence):
    def __init__(
        self,
        BATCH_SIZE,
        phase="train",
        train_val_split=None,
        seed=None,
        return_kp=False,
    ):
        if phase == "train":
            self.IMG_DIR = config.IMG_DIR_TRAIN
            self.ANNO_FILE = config.ANNO_FILE_TRAIN

        elif phase == "val":
            self.IMG_DIR = config.IMG_DIR_VAL
            self.ANNO_FILE = config.ANNO_FILE_VAL

        elif phase == "trainval" or phase == "valtrain":
            assert train_val_split > 0 and train_val_split < 1
            self.IMG_DIR = config.IMG_DIR_TRAIN_VAL
            self.ANNO_FILE = config.ANNO_FILE_TRAIN_VAL

        else:
            self.IMG_DIR = config.IMG_DIR_TEST
            self.ANNO_FILE = config.ANNO_FILE_TEST

        self.phase = phase
        self.BATCH_SIZE = BATCH_SIZE
        self.train_val_split = train_val_split
        self.coco = COCO(self.ANNO_FILE)
        self.return_kp = return_kp

        # to be used if single json file available
        # self.img_ids = list(self.coco.imgs.keys())
        # function to update valid image ids -self.valid_img_ids
        # self.get_valid_img_ids()

        # to be used if separate json files available for train/test/val/trainval
        self.valid_img_ids = list(self.coco.imgs.keys())

        self.datasetlen = len(self.valid_img_ids)
        self.id = 0
        if (train_val_split and phase == "trainval") or (
            train_val_split and phase == "valtrain"
        ):
            self.train_len = int(np.ceil(self.datasetlen * self.train_val_split))
            self.val_len = int(self.datasetlen) - self.train_len
            self.img_ids_train = self.valid_img_ids[: self.train_len]
            self.img_ids_val = self.valid_img_ids[self.train_len :]
        else:
            self.train_len = int(self.datasetlen)
            self.val_len = int(self.datasetlen)
            self.img_ids_train = self.valid_img_ids
            self.img_ids_val = self.valid_img_ids

        if self.phase == "train" or self.phase == "trainval":
            self.datasetlen = self.train_len
        elif self.phase == "val" or self.phase == "valtrain":
            self.datasetlen = self.val_len
        else:
            self.datasetlen = len(self.valid_img_ids)

        self.seed = seed

    def __len__(self):
        return math.ceil(self.datasetlen // self.BATCH_SIZE)

    def on_epoch_end(self):
        def seed_fun():
            if self.seed is None:
                seed = np.random.randint(1e6)
            else:
                seed = self.seed
            return seed

        # Shuffle dataset for next epoch
        random.shuffle(self.img_ids_train)
        random.shuffle(self.img_ids_val)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

    def get_valid_img_ids(self):
        self.valid_img_ids = []

        for img_id in self.img_ids:
            check_flag = []
            filepath_flag = False
            img_anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            # sometimes image id is not present in SSS annotation
            image_filename = self.coco.imgs[img_id]["file_name"]
            clean_imagename = re.sub(r"[^a-zA-Z0-9._-]+", "_", image_filename).replace(
                "__", "_"
            )
            clean_imagename = clean_imagename.replace("-", "_")
            for anno in img_anns:
                check_flag = anno["keypoints"]
                break

            if self.phase == "train":
                SSS_data_filepath_iter = SSS_data_filepath[0:1]
            elif self.phase == "val":
                SSS_data_filepath_iter = SSS_data_filepath[1:2]
            elif self.phase == "trainval" or self.phase == "valtrain":
                SSS_data_filepath_iter = SSS_data_filepath[2:3]
            else:
                SSS_data_filepath_iter = SSS_data_filepath[3:4]

            for img_dir in SSS_data_filepath_iter:
                filepath_check = os.path.join(img_dir, clean_imagename)

                if os.path.exists(filepath_check):
                    filepath_flag = True

            if len(check_flag) > 0 and filepath_flag:
                self.valid_img_ids.append(img_id)

    def read_data(self, filepath_list, img_id_list):
        filepath_list = filepath_list.numpy()
        img_id_list = img_id_list.numpy()

        img_batch = np.zeros(
            (
                config.BATCH_SIZE,
                config.IMAGE_SHAPE[0],
                config.IMAGE_SHAPE[1],
                3,
            )
        )
        kp_batch = np.zeros(
            (
                config.BATCH_SIZE,
                config.IMAGE_SHAPE[0],
                config.IMAGE_SHAPE[1],
                config.NUM_KP + 1,
            )
        )
        short_offsets_batch = np.zeros(
            (
                config.BATCH_SIZE,
                config.IMAGE_SHAPE[0],
                config.IMAGE_SHAPE[1],
                config.NUM_KP * 2,
            )
        )
        mid_offsets_batch = np.zeros(
            (
                config.BATCH_SIZE,
                config.IMAGE_SHAPE[0],
                config.IMAGE_SHAPE[1],
                NUM_EDGES * 4,
            )
        )

        for i, (filepath, img_id) in enumerate(zip(filepath_list, img_id_list)):
            img = cv2.imread(filepath.decode())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            keypoints = []
            area_list = []
            img_anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

            for anno in img_anns:
                keypoints.append([int((x)) for x in anno["keypoints"]])
                area_list.append(anno["area"])

            # code added for correction in annotation for data at end points of image
            h, w, c = img.shape
            for batch in range(len(keypoints)):
                for idx0, val in enumerate(keypoints[batch]):
                    idx = idx0 + 1
                    if (idx % 3 == 1 and val == w) or (idx % 3 == 2 and val == h):
                        keypoints[batch][idx0] = keypoints[batch][idx0] - 1
            # code end

            kp = np.reshape(keypoints, (-1, config.NUM_KP, 3))

            # do augmentation only when it is training else only do resizing
            if self.phase == "train":
                img, kp = Augmentation.augment(img, kp, is_aug=True)
            else:
                img, kp = Augmentation.augment(img, kp, is_aug=False)

            kp = [np.squeeze(k) for k in np.split(kp, kp.shape[0], axis=0)]
            kp_maps, short_offsets, mid_offsets = get_ground_truth(kp)
            # normalization for input images
            img = Augmentation.preprocess(img)

            keypoints_flattened = np.array(keypoints).flatten()
            kp_dummy = np.zeros((config.IMAGE_SHAPE[0] * config.IMAGE_SHAPE[1],)) - 1
            kp_dummy[0 : len(keypoints_flattened)] = keypoints_flattened
            kp_dummy = kp_dummy.reshape((config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]))

            img_batch[i] = img
            kp_batch[i, :, :, 0 : config.NUM_KP] = kp_maps
            kp_batch[i, :, :, config.NUM_KP] = kp_dummy
            short_offsets_batch[i] = short_offsets
            mid_offsets_batch[i] = mid_offsets
            return img_batch, kp_batch, short_offsets_batch, mid_offsets_batch

    def iterate_data(self, filepath_list, img_id_list):
        img_shape = (None, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 3)
        kp_batch_shape = (
            None,
            config.IMAGE_SHAPE[0],
            config.IMAGE_SHAPE[1],
            config.NUM_KP + 1,
        )
        short_offsets_batch_shape = (
            None,
            config.IMAGE_SHAPE[0],
            config.IMAGE_SHAPE[1],
            config.NUM_KP * 2,
        )
        mid_offsets_batch_shape = (
            None,
            config.IMAGE_SHAPE[0],
            config.IMAGE_SHAPE[1],
            NUM_EDGES * 4,
        )

        [img_batch, kp_batch, short_offsets_batch, mid_offsets_batch] = tf.py_function(
            self.read_data,
            [filepath_list, img_id_list],
            [tf.float32, tf.float32, tf.float32, tf.float32],
        )

        img_batch.set_shape(img_shape)
        kp_batch.set_shape(kp_batch_shape)
        short_offsets_batch.set_shape(short_offsets_batch_shape)
        mid_offsets_batch.set_shape(mid_offsets_batch_shape)

        return img_batch, (kp_batch, short_offsets_batch, mid_offsets_batch)

    def get_batch(self, i):
        h, w, c = config.IMAGE_SHAPE
        imgs_names_batch = []
        img_ids_batch = []

        if self.phase == "train" or self.phase == "trainval":
            give_id_list = self.img_ids_train[
                i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE
            ]

        elif self.phase == "val" or self.phase == "valtrain":
            give_id_list = self.img_ids_val[
                i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE
            ]

        else:
            give_id_list = self.valid_img_ids[
                i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE
            ]

        for j, img_id in enumerate(give_id_list):
            image_filename = self.coco.imgs[img_id]["file_name"]
            clean_imagename = re.sub(r"[^a-zA-Z0-9._-]+", "_", image_filename).replace(
                "__", "_"
            )
            clean_imagename = clean_imagename.replace("-", "_")
            filepath = os.path.join(self.IMG_DIR, clean_imagename)
            imgs_names_batch.append(filepath)
            img_ids_batch.append(img_id)

        return [imgs_names_batch, img_ids_batch]

    def __getitem__(self, index):
        h, w, c = config.IMAGE_SHAPE

        imgs_names_batch, img_ids_batch = self.get_batch(index)

        return imgs_names_batch, img_ids_batch


def __main__():
    # initialize generator object
    train_generator = DataGenerator(5, "train", 0.8, seed=0.5)
    val_generator = DataGenerator(5, "val", 0.8, seed=0.5)

    train_ds = tf.data.Dataset.from_generator(
        train_generator,
        output_types=(tf.float32, (tf.float32, tf.float32, tf.float32)),
        output_shapes=(
            (None, img_shape[0], img_shape[1], 3),
            (
                (None, img_shape[0], img_shape[1], 17),
                (None, img_shape[0], img_shape[1], 34),
                (None, img_shape[0], img_shape[1], 64),
            ),
        ),
    )

    validation_ds = tf.data.Dataset.from_generator(
        val_generator,
        output_types=(tf.float32, (tf.float32, tf.float32, tf.float32)),
        output_shapes=(
            (None, img_shape[0], img_shape[1], 3),
            (
                (None, img_shape[0], img_shape[1], 17),
                (None, img_shape[0], img_shape[1], 34),
                (None, img_shape[0], img_shape[1], 64),
            ),
        ),
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)

    train_generator_tfds = DataGeneratorTFDS(config.BATCH_SIZE, "train", 0.8, seed=0.5)
    train_tfds = tf.data.Dataset.from_generator(
        train_generator_tfds,
        output_types=(tf.string, tf.int32),
        output_shapes=((None,), (None,)),
    )
    train_tfds_aug = train_tfds.map(
        train_generator_tfds.iterate_data, num_parallel_calls=tf.data.AUTOTUNE
    )

    for ims in train_tfds_aug:
        import pdb

        pdb.set_trace()

    print("No of train samples", train_generator.train_len)
    print(
        "No of train samples",
        len(train_generator) * train_generator.BATCH_SIZE,
    )
    print("No of val samples", val_generator.val_len)
    print("total samples", train_generator.datasetlen)

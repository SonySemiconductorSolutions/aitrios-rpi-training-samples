import cv2
from third_party.keraspersonlab.data_parser import map_SSS_to_personlab
import albumentations as A
from albumentations import (
    Resize,
    RandomBrightnessContrast,
    HueSaturationValue,
    CenterCrop,
    CoarseDropout,
    RandomSunFlare,
    Perspective,
)
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate

NORM_FACTOR = None

config = None


def set_config(config_i):
    global config
    global NORM_FACTOR

    config = config_i
    NORM_FACTOR = config.NORM_FACTOR


class Augmentation:
    @staticmethod
    def preprocess(img):
        return img / NORM_FACTOR

    @staticmethod
    def augment(img=None, keypoints=None, is_aug=True):
        keypoints_orig = map_SSS_to_personlab(keypoints)
        keypoints = keypoints_orig.reshape(1, -1, 3)
        # for batch in range(keypoints.shape[0]):
        batch = 0
        original_points = keypoints.copy()[batch, :, 0:2]

        transformed_reordered_points = []
        aug_flip = False

        class_labels_orig = list(config.KEYPOINTS_DICT.values())
        for n in range(0, keypoints_orig.shape[0]):
            if n > 0:
                class_labels = class_labels + [
                    x + "_" + str(n) for x in class_labels_orig
                ]
            else:
                class_labels = class_labels_orig

        if is_aug:
            transform = A.ReplayCompose(
                [
                    # be careful with rotate, keypoints generated needs to be postprocessed as in flip
                    # Rotate(limit=10,p=1),
                    ShiftScaleRotate(
                        p=0.95,
                        shift_limit=0.1,
                        scale_limit=(-0.7, 0.2),
                        rotate_limit=(-180, 180),
                        border_mode=cv2.BORDER_CONSTANT,
                    ),
                    Perspective(
                        p=0.5,
                    ),
                    CenterCrop(p=0.5, height=240, width=320),
                    RandomBrightnessContrast(p=0.2),
                    RandomSunFlare(
                        p=0.5,
                    ),
                    CoarseDropout(
                        p=0.5, 
                        max_holes=2,
                        max_height=0.2,
                        min_height=0.1,
                        max_width=0.2,
                        min_width=0.1,
                        mask_fill_value=0,
                    ),
                    HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.5,
                    ),
                    Resize(
                        height=config.IN_HEIGHT,
                        width=config.IN_WIDTH,
                        interpolation=cv2.INTER_AREA,
                    ),
                ],
                keypoint_params=A.KeypointParams(
                    format="xy",
                    label_fields=["class_labels"],
                    remove_invisible=True,
                ),
            )
        else:
            transform = A.ReplayCompose(
                [
                    Resize(
                        height=config.IN_HEIGHT,
                        width=config.IN_WIDTH,
                        interpolation=cv2.INTER_AREA,
                    )
                ],
                keypoint_params=A.KeypointParams(
                    format="xy",
                    label_fields=["class_labels"],
                    remove_invisible=True,
                ),
            )

        transformed = transform(
            image=img, keypoints=original_points, class_labels=class_labels
        )
        transformed_image = transformed["image"]
        transformed_keypoints = transformed["keypoints"]
        transformed_class_labels = transformed["class_labels"]

        for i, class_name in enumerate(class_labels):
            if class_name in transformed_class_labels:
                indx = transformed_class_labels.index(class_name)
                if (
                    original_points[i][0] == 0
                    and original_points[i][1] == 0
                    and keypoints[batch, :, :][i][2] == 0
                ):
                    transformed_reordered_points.append((0, 0, 0))
                else:
                    transformed_reordered_points.append(
                        (
                            transformed_keypoints[indx][0],
                            transformed_keypoints[indx][1],
                            2,
                        )
                    )
            else:
                transformed_reordered_points.append((0, 0, 0))

        keypoints[batch, :, :] = transformed_reordered_points

        for method_dict in transformed["replay"]["transforms"]:
            if method_dict["__class_fullname__"] == "HorizontalFlip":
                aug_flip = method_dict["applied"]
                # print("flipped*******",aug_flip)

        if aug_flip:
            tmpLeft = keypoints[batch, config.LEFT_KP, :]
            tmpRight = keypoints[batch, config.RIGHT_KP, :]
            keypoints[batch, config.LEFT_KP, :] = tmpRight
            keypoints[batch, config.RIGHT_KP, :] = tmpLeft

        keypoints = keypoints.reshape(-1, config.NUM_KP, 3)
        return transformed_image, keypoints

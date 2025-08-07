
# New Dataset of imx500_zoo PoseNet 

## Introduction

To use a new dataset with PoseNet, you need the dataset files and to modify the configuration files.
The new dataset has new KEY_POINTS and new EDGES.
The path to the dataset has changed.

## Files to Modify
```
mandatory
 1 JSON files of dataset
 2 INI file
 3 JSON file

optional
 4 augmentation.py
 5 posenet_trainer.py
```

### Mandatory Modifications

1. JSON files of dataset in samples/data/

In the "categories" section :

- Change "supercategory" and "name" to the new names.
- Update "keypoints" to the new KEY_POINTS in the correct order.
- Update "skeleton" to the new EDGES in the correct order, starting from 0.

```json
    "categories": [
        {
            "supercategory": "arrow",
            "id": 1,
            "name": "arrow",
            "keypoints": [
                "Left",
                "Middle left",
                "Bottom left",
                "Bottom right",
                "Middle right",
                "Right",
                "Top"
            ],
            "skeleton": [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7]
            ]
        }
    ]
```

In the "annotations" section :

- Change "num_keypoints" to the number of KEY_POINTS.
- Update "keypoints" to the coordinates of each KEY_POINT in the correct order. Each coordinate consists of x, y, and v, where v is a visibility flag (v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible).
```json
    "annotations": [
        {
            "num_keypoints": 7,
            "category_id": 1,
            "image_id": 0,
            "id": 147,
            "keypoints": [
                333.4, 287.2, 2,
                315.9, 286.9, 2,
                314.9, 323.6, 2,
                277.4, 322.9, 2,
                278.5, 285.7, 2,
                261.1, 285.1, 2,
                298, 250.2, 2
            ],
            "area": 2695
        },
```

2. INI file in samples/

- Change NUM_CLASSES in the [MODEL] section to the number of KEY_POINTS.
```ini
NUM_CLASSES = 7
```

- Update CONFIG in the [TRAINER] section to the path of the JSON file.
```ini
CONFIG = ./config/posenet_arrow.json
```

3. JSON file in samples/config/

In the "DATASET" section :

- Change "TYPE" to the new name.
- Update "NUM_KP" to the number of KEY_POINTS.
- Update "KEYPOINTS" to the names of the KEY_POINTS in the correct order.
- Update "KEYPOINTS_DICT" to the pairs of numbers and names of the KEY_POINTS in the correct order.
- Update "RIGHT_KP" to the names of the right KEY_POINTS in the correct order. "LEFT_KP" should be updated similarly for the left KEY_POINTS. Blank is [].
- Update "EDGES" to the pairs of KEY_POINTS starting from 0 as EDGES.
- Update "kpt_oks_sigmas" to 1 / number of KEY_POINTS. Weighting is acceptable.

```json
    "DATASET": {
        "TYPE": "ARROW",
        "NUM_KP": 7,
        "KEYPOINTS": [
            "Left",
            "Middle left",
            "Bottom left",
            "Bottom right",
            "Middle right",
            "Right",
            "Top"
        ],
        "KEYPOINTS_DICT": {
            "0": "Left",
            "1": "Middle left",
            "2": "Bottom left",
            "3": "Bottom right",
            "4": "Middle right",
            "5": "Right",
            "6": "Top"
        },
        "RIGHT_KP": [5, 4, 3],
        "LEFT_KP":  [0, 1, 2],
        "EDGES": [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 0]
        ],
        "NORM_FACTOR": 256.0,
        "KP_RADIUS": 32,
        "PEAK_THRESH": 0.004,
        "OKS_THRESH": 0.5,
        "kpt_oks_sigmas": [
            0.14,
            0.14,
            0.14,
            0.14,
            0.14,
            0.14,
            0.14
        ],
        "NMS_THRESH": 32
    },
```

- Update each path to the new dataset files.
```json
   "PATH": {
        "ANNO_FILE_TRAIN": "data/ArrowPosenet/arrow/arrow_train.json",
        "IMG_DIR_TRAIN":   "data/ArrowPosenet/arrow/train",
        "ANNO_FILE_VAL":   "data/ArrowPosenet/arrow/arrow_val.json",
        "IMG_DIR_VAL":     "data/ArrowPosenet/arrow/val",
        "ANNO_FILE_TEST":  "data/ArrowPosenet/arrow/arrow_val.json",
        "IMG_DIR_TEST":    "data/ArrowPosenet/arrow/val"
    },
```

- Update "CONF_MAP_KEY" to the first name of the KEY_POINTS.
```json
    "VISUALISE": {
        "CONF_MAP_KEY": "Top"
    }
```

### Optional Modifications

4. src/imx500_zoo/utilities/posenet/utils/augmentation.py

The [Augmentation setting](https://albumentations.ai/docs/getting_started/transforms_and_targets/) can be modified.

ShiftScaleRotate and CenterCrop are very strong transformations and can be removed if necessary.
```python
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
                    CenterCrop(p=0.5, height=240, width=320),
                    RandomBrightnessContrast(p=0.2),
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
``` 

5. src/imx500_zoo/trainers/posenet_trainer.py

The value of [patience](https://keras.io/api/callbacks/early_stopping/) can be modified.
```python
    earlystop_callback = EarlyStopping(
        monitor="val_{}".format(monitor), patience=10, verbose=1, mode="min"
    )
```


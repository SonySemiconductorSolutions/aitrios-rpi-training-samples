import torch
import numpy as np
import cv2


class AlbuImageFolder(torch.utils.data.Dataset):
    def __init__(self, img_paths, img_bboxes, category_ids, transform):
        self.img_paths = img_paths
        self.img_bboxes = img_bboxes
        self.category_ids = category_ids
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_bbox = []
        img_bbox.append(self.img_bboxes[index])
        category_ids = []
        category_ids.append(self.category_ids[index])

        img = cv2.imread(img_path)

        result = self.transform(
            image=np.array(img), bboxes=img_bbox, class_labels=category_ids
        )
        result_image = result["image"]
        result_bboxes = result["bboxes"][0]
        result_cids = result["class_labels"][0]

        return result_image, result_bboxes, result_cids

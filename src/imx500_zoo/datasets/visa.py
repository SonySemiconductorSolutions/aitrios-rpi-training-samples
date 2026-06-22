import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import glob
from imx500_zoo import utilities
import subprocess
import numpy as np
import random


DOWNLOAD_VISA_DATASET = (
    "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/"
    "VisA_20220922.tar"
)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class VisA:
    def __init__(self, config):
        self.config = config

    def download_and_extract_dataset(self, data_path):
        utilities.download_and_extract_tar(
            DOWNLOAD_VISA_DATASET, data_path, folder_name="VisA_20220922")

    def prepare_data_split(self, data_path):
        visa_data_folder = os.path.abspath(os.path.join(
            data_path, "VisA_20220922"))
        visa_output_folder = os.path.abspath(os.path.join(
            data_path, "VisA_pytorch"))

        if os.path.exists(visa_output_folder):
            print(
                f"{visa_output_folder} already exists, "
                "skipping data convert preparation."
            )
            return

        spot_diff_dir = os.path.abspath(os.path.join(
            data_path, "../../../third_party/spot-diff"))
        if not os.path.exists(spot_diff_dir):
            print("Cloning spot-diff repository...")
            subprocess.run(
                ["git", "clone",
                 "https://github.com/amazon-science/spot-diff.git",
                 spot_diff_dir], check=True)

        print("Preparing data convert...")

        split_script = os.path.join(spot_diff_dir, "utils/prepare_data.py")
        split_file = os.path.join(spot_diff_dir, "split_csv/1cls.csv")

        command = [
            "python", split_script,
            "--split-type", "1cls",
            "--data-folder", visa_data_folder,
            "--save-folder", visa_output_folder,
            "--split-file", split_file
        ]

        subprocess.run(command, check=True)

    def setup(self):
        base_path = self.config["PATH"]["DATA"]

        target_object = self.config["DATASET"].get("OBJECT", "pipe_fryum")
        print("VisaDataset setting for object:", target_object)

        self.download_and_extract_dataset(base_path)
        self.prepare_data_split(base_path)

        input_size = int(self.config["MODEL"].get("INPUT_SIZE", 256))
        model_mean = [0.485, 0.456, 0.406]
        model_std = [0.229, 0.224, 0.225]

        self.data_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.CenterCrop(input_size),
            transforms.Normalize(mean=model_mean, std=model_std)
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()
        ])

        batch_size = int(self.config["TRAINER"]["BATCH_SIZE"])
        num_workers = int(self.config["TRAINER"]["NUM_WORKERS"])

        dataset_path = os.path.join(base_path, "VisA_pytorch", "1cls")
        train_path = os.path.join(dataset_path, target_object, "train")
        test_path = os.path.join(dataset_path, target_object)

        self.dataset_train = ImageFolder(root=train_path,
                                         transform=self.data_transform)
        self.dataset_valid = MVTecDataset(root=test_path,
                                          transform=self.data_transform,
                                          gt_transform=self.gt_transform,
                                          phase="test")

        self.trainloader = DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn
        )

        self.validloader = DataLoader(
            self.dataset_valid,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn
        )

    def get_loaders(self):
        return self.trainloader, self.validloader, self.validloader

    def get_single_data(self, index):
        return self.dataset_valid[index]


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_paths, self.gt_paths, self.labels, self.types = (
            self.load_dataset()
        )

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(
                    self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(
                    self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(
                    self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(
            gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[
            idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type

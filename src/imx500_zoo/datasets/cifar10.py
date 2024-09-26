import os
import torch
import torchvision
from torchvision import transforms
from imx500_zoo.utilities import conv


class Cifar10:
    def __init__(self, config):
        self.config = config

    def setup(self):
        print("Dataset setting")
        input_size = int(self.config["MODEL"]["INPUT_SIZE"])

        self.model_mean = conv.list(self.config["MODEL"]["MEAN"])
        self.model_std = conv.list(self.config["MODEL"]["STD"])

        self.data_path = self.config["PATH"]["DATA"]
        self._set_dataset_path()
        self.dataset = torchvision.datasets.CIFAR10
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.model_mean, std=self.model_std),
            ]
        )

        batch_size = int(self.config["TRAINER"]["BATCH_SIZE"])
        num_workers = int(self.config["TRAINER"]["NUM_WORKERS"])

        self.dataset_train = self.dataset(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=True,
        )
        self.dataset_valid = self.dataset(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False,
        )

        self.trainloader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
        )

        self.validloader = torch.utils.data.DataLoader(
            self.dataset_valid,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
        )

        self.imageloader = self.validloader

    def _set_dataset_path(self):
        self.config["PATH"]["DATASET"] = os.path.join(
            self.data_path, "cifar-10-batches-py"
        )

    def get_loaders(self):
        return self.trainloader, self.validloader, self.validloader

    def get_single_data(self, index):
        return self.dataset_valid[index]

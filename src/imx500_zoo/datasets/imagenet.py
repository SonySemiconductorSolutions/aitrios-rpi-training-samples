from imx500_zoo import utilities
from imx500_zoo.utilities import conv
import os
from torchvision import transforms
import torchvision
import torch

DOWNLOAD_DATASET = (
    r"https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
)

DOWNLOAD_DEVKIT = (
    r"https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
)


class ImageNet:
    def __init__(self, config):
        self.config = config

    def download(self, data_path):
        utilities.download_file(
            DOWNLOAD_DATASET,
            data_path,
            "ILSVRC2012_img_val",
        )
        utilities.download_file(
            DOWNLOAD_DEVKIT,
            data_path,
            "ILSVRC2012_devkit_t12",
        )

    def setup(self):
        input_size = int(self.config["MODEL"]["INPUT_SIZE"])
        self.data_path = self.config["PATH"]["DATA"]
        self.download(self.data_path)
        self._set_dataset_path()
        self.batch_size = int(self.config["QUANTIZER"]["BATCH_SIZE"])
        self.num_workers = int(self.config["QUANTIZER"]["NUM_WORKERS"])

        self.model_mean = conv.list(self.config["MODEL"]["MEAN"])
        self.model_std = conv.list(self.config["MODEL"]["STD"])

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.model_mean, std=self.model_std),
            ]
        )

        self.dataset_valid = torchvision.datasets.ImageNet(
            root=f"{self.data_path}/", split="val", transform=self.transform
        )

        self.validloader = torch.utils.data.DataLoader(
            self.dataset_valid,
            batch_size=self.batch_size,
            drop_last=True,
        )

    def _set_dataset_path(self):
        self.config["PATH"]["DATASET"] = os.path.join(self.data_path, "val")

    def get_loaders(self):
        return None, None, self.validloader

    def get_single_data(self, index):
        return self.datase_valid[index]

import torch
import torchvision
import os
from imx500_zoo import utilities
from torchvision import transforms


DOWNLOAD_DATASET = (
    r"https://github.com/SonySemiconductorSolutions/aitrios-rpi-dataset-sample/raw/refs/heads/main/card_classification.zip"
)

class CardClassification:
    def __init__(self, config):
        self.config = config

    def download(self, data_path):
        utilities.download_zip(
            DOWNLOAD_DATASET,
            data_path,
            "train_data",
            target_name="card_classification.zip"
        )

    def setup(self):
        self.data_path = "./data/card_classification"
        self.train_path = os.path.join(self.data_path, "train_data")
        self.valid_path = os.path.join(self.data_path, "valid_data")
        self.download(self.data_path)
        self._set_dataset_path()

        self.setup_transform()

        batch_size = int(self.config["TRAINER"]["BATCH_SIZE"])
        num_workers = int(self.config["TRAINER"]["NUM_WORKERS"])

        self.dataset_train = torchvision.datasets.ImageFolder(
            root=self.train_path, transform=self.transform["train"]
        )
        self.dataset_valid = torchvision.datasets.ImageFolder(
            root=self.valid_path, transform=self.transform["valid"]
        )

        self.trainloader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=num_workers,
        )

        self.validloader = torch.utils.data.DataLoader(
            self.dataset_valid,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=num_workers,
        )

        self.imageloader = self.validloader

    def _set_dataset_path(self):
        self.config["PATH"]["DATASET"] = self.train_path

    def setup_transform(self):
        input_width_size = int(self.config["MODEL"]["INPUT_SIZE"])
        input_height_size = int(input_width_size * 0.75)
        pad_width = int(input_width_size / 4 / 2)

        self.transform = {
            "train": transforms.Compose(
                [
                    transforms.Resize((input_width_size, input_height_size)),
                    transforms.Pad(  # pad to resize to 224x224
                        (pad_width, 0, pad_width, 0), padding_mode="edge"
                    ),
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomAffine(
                        degrees=[-60, 60],
                        translate=(0.0, 0.0),
                        scale=(1.0, 2.0),
                    ),
                    transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
                    transforms.RandomRotation(degrees=60),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            "valid": transforms.Compose(
                [
                    transforms.Resize((input_width_size, input_height_size)),
                    transforms.Pad(  # pad to resize to 224x224
                        (pad_width, 0, pad_width, 0), padding_mode="edge"
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        }

    def get_loaders(self):
        return self.trainloader, self.validloader, self.imageloader

    def get_single_data(self, index):
        return self.dataset_valid[index]

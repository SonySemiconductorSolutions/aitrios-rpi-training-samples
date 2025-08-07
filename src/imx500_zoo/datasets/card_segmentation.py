from imx500_zoo.utilities.deeplab_v3p.segmentation import Segmentation
from imx500_zoo.quantizers.mct_keras import get_representative_dataset
from imx500_zoo.quantizers.mct_keras_deeplab import CustomDataset
from imx500_zoo import utilities
import os

DOWNLOAD_DATASET = r"https://github.com/SonySemiconductorSolutions/aitrios-rpi-dataset-sample/raw/main/card_segmentation.zip"

class CardSegmentation:
    def __init__(self, config):
        self.ini = config

        self.trainloader = None
        self.validloader = None
        self.dataloader_quant = None
        self.dataloader_eval = None

    def setup(self):
        self.config = self.ini.deeplab.config
        self.nclass = self.config.n_classes
        self.batchsize = self.config.batch_size
        self.imagesize = self.config.image_size
        self.d_root = self.config.data_rootpath

        self._download_dataset()
        self._setup_train()
        self._setup_quant()
        self._setup_valid()

    def get_loaders(self):
        return (
            self.trainloader,
            self.validloader,
            self.dataloader_quant,
            self.dataloader_eval,
        )

    def _download_dataset(self):
        d_exist = os.path.join(self.config.data_rootpath, "train")
        if os.path.isdir(d_exist):
            print(f"skip download dataset : {d_exist}")
        else:
            utilities.download_zip(
                url=DOWNLOAD_DATASET,
                data_path=self.config.data_rootpath,
                exist_path="train",
                target_name="card_segmentation.zip",
            )

    def _setup_train(self):
        # Define the train and valid data generator.
        self.trainloader = Segmentation(
            config=self.config,
            dir=self.d_root,
            batch_size=self.batchsize,
            resize_shape=self.imagesize,
            blur=5,
            crop_shape=None,
            mode="train",
            n_classes=self.nclass,
            h_flip_en=True,
            v_flip_en=False,
            brightness=0.3,
            rotation=180,
            zoom=0.1,
            seed=7,
            contrast_en=False,
        )

        self.validloader = Segmentation(
            config=self.config,
            dir=self.d_root,
            batch_size=self.batchsize,
            resize_shape=self.imagesize,
            blur=0,
            crop_shape=None,
            mode="valid",
            n_classes=self.nclass,
            h_flip_en=True,
            v_flip_en=False,
            brightness=0.1,
            rotation=False,
            zoom=0.05,
            seed=7,
            contrast_en=False,
        )

    def _setup_quant(self):
        print(f"load dataset of quant : {self.config.rep_dataset}")
        dataset = CustomDataset(
            base_path=self.config.rep_dataset,
            img_size=self.config.image_width,
        )
        self.dataloader_quant = get_representative_dataset(dataset, is_shuffle=True)

    def _setup_valid(self):
        self.dataloader_eval = Segmentation(
            config=self.config,
            dir=self.d_root,
            batch_size=1,
            resize_shape=self.imagesize,
            crop_shape=None,
            mode="test",
            n_classes=self.nclass,
            h_flip_en=False,
            v_flip_en=False,
            brightness=0,
            rotation=False,
            zoom=0,
            seed=7,
            contrast_en=False,
        )

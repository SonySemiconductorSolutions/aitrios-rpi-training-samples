from imx500_zoo import utilities
from imx500_zoo.quantizers.mct_tflite import MCTWorkFlow
from imx500_zoo.utilities.posenet.data_generator import (
    DataGenerator,
    DataGenType,
)
from imx500_zoo.validators.keras_posenet_validator import BATCH_SIZE as BS

class DefaultPosenet:
    DOWNLOAD_DATASET = None
    ZIP_SUBFOLDER = None
    ZIP_FILENAME = None
    
    def __init__(self, config):
        self.config = config
        self.data_path = self.config["PATH"]["DATA"]

        self.trainloader = None
        self.validloader = None
        self.dataloader_quant = None
        self.dataloader_eval = None

    def setup(self):
        print(f"dataset path: {self.data_path}")
        self._download()
        self._setup_dataloder()

    def _download(self):
        if self.DOWNLOAD_DATASET is None:
            print(f"No download dataset provided for {self.__class__.__name__}, Skipping download.")
            return
        if self.ZIP_FILENAME is None:
            raise ValueError(f"ZIP_FILENAME is not set for {self.__class__.__name__}")
        if self.ZIP_SUBFOLDER is None:
            raise ValueError(f"ZIP_SUBFOLDER is not set for {self.__class__.__name__}")
        
        utilities.download_zip(
            self.DOWNLOAD_DATASET,
            self.data_path,
            self.ZIP_SUBFOLDER,
            target_name=self.ZIP_FILENAME,
        )

    def _setup_dataloder(self):
        config = self.config.posenet
        BATCH_SIZE = config.BATCH_SIZE
        self.trainloader = DataGenerator(BATCH_SIZE, "train", 0.8, seed=0.5)
        self.validloader = DataGenerator(BATCH_SIZE, "val", 0.8, seed=0.5)

        ext = MCTWorkFlow.tflite()
        is_h5 = ext[-2:] != "h5"
        batch_size = 4 if is_h5 else 1
        self.dataloader_quant = DataGenerator(
            phase="test",
            BATCH_SIZE=batch_size,
            train_val_split=0.8,
            seed=0.5,
            return_kp=True,
            testDataFldr=config.IMG_DIR_TEST,
            annotation_file=config.ANNO_FILE_TEST,
            dg_type=DataGenType.QUANT,
        )

        self.dataloader_eval = DataGenerator(BS, "test", 0.8, seed=0.5, return_kp=True)

    def get_loaders(self):
        return (
            self.trainloader,
            self.validloader,
            self.dataloader_quant,
            self.dataloader_eval,
        )

import os
import cv2
import copy
import numpy as np
from PIL import Image

from imx500_zoo.quantizers.mct_keras import MctKerasBase
from third_party.kerasdeeplabv3plus.deeplabv3 import DeepLabArchitecture

class CustomDataset:
    def __init__(
        self, base_path="your_dataset_directory", img_size=0, batch_size=1
    ):
        """
        Initializes the dataset class to handle image loading and
        preprocessing.

        Args:
            base_path (str): Path to the directory containing the images.
            img_size (int): The target size to resize images.
            batch_size (int): Number of images per batch.
        """
        # Get all image file paths in the directory
        self.images_names = [
            os.path.join(base_path, fname)
            for fname in os.listdir(base_path)
            if fname.endswith(".jpg") or fname.endswith(".png")
        ]
        self.inds = list(range(len(self.images_names)))
        self.img_size = img_size
        self.batch_size = batch_size

    def shuffle(self):
        """Shuffle the order of the images."""
        self.inds = np.random.permutation(self.inds)

    def __len__(self):
        """Calculate the total number of batches."""
        return int(np.ceil(len(self.images_names) / self.batch_size))

    def __iter__(self):
        """Generator for yielding batches of images."""
        img_batch = []
        for i in self.inds:
            img_name = self.images_names[i]
            img = np.array(Image.open(img_name))
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
            img_batch.append(img)

            # Yield a batch of images when it reaches the batch size
            if len(img_batch) == self.batch_size:
                yield np.stack(img_batch)
                img_batch = []

        # Yield any remaining images as the final batch
        if len(img_batch) > 0:
            yield np.stack(img_batch)

class MctKerasDeeplab(MctKerasBase):
    def quantize(self, model, dataloader_quant):
        dlab = self.ini.deeplab
        config = dlab.config
        self.setup_gpu()
        if model is None:
            model = self.load_model()
        loader = copy.deepcopy(dataloader_quant)
        mct_model = self.run_mct(model, loader)
        self.save(mct_model, config.mct_model, config.PATH_QUANTIZED_KERAS_SUMMARY)
    
    def load_model(self):
        config = self.ini.deeplab.config

        model_path = config.best_weights
        input_size = (
            config.image_width,
            config.image_height,
            3,
        )
        arc = DeepLabArchitecture()
        model = arc.Deeplabv3(input_shape=input_size, classes=config.n_classes)
        print(f"load weights : {model_path}")
        model.load_weights(model_path)

        return model
    

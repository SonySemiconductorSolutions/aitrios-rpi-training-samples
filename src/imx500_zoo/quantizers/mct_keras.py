import model_compression_toolkit as mct
import torch

from typing import Iterator, Tuple, List
from imx500_zoo.utilities import tf_utility

class MctKeras:
    def __init__(self, config):
        # configuration
        self.config = config

    def _get_representative_dataset(self, n_iter: int, dataset_loader: Iterator[Tuple]):
        """
        This function creates a representative dataset generator.

        Args:
        n_iter: number of iterations for MCT to calibrate on
        Returns:
        A representative dataset generator
        """

        def representative_data_gen() -> Iterator[List]:
            """
            Creates a representative dataset generator from a PyTorch data loader, The
            generator yields numpy
            arrays of batches of shape: [Batch, H, W ,C].

            Returns:
            A representative dataset generator
            """
            ds_iter = iter(dataset_loader)
            for _ in range(n_iter):
                yield [next(ds_iter)[0]]

        return representative_data_gen

    def quantize(self, model, dataloader):
        n_iter = int(self.config["QUANTIZER"]["CALIB_ITERATIONS"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        tpc = mct.get_target_platform_capabilities("tensorflow", "imx500")
        quantized_model = None

        # Preform post training quantization
        quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(
            model,
            representative_data_gen=self._get_representative_dataset(
                n_iter, dataloader
            ),
            target_platform_capabilities=tpc,
        )

        print("Quantized model is ready")

        # export model
        target_path = self.config["PATH"]["QUANTIZED_KERAS"]
        print(f"saved {target_path}")
        quantized_model.save(target_path)

        return quantized_model


def get_representative_dataset(dataset, is_shuffle=False):
    """
    Creates a representative dataset generator that yields batches of images.

    Returns:
        A representative dataset generator.
    """
    if is_shuffle:
        dataset.shuffle()

    def _representative_dataset() -> Iterator[List]:
        ds_iter = iter(dataset)
        for batch in ds_iter:
            yield [batch]  # Yield batches of images

    return _representative_dataset

class MctKerasBase:
    def __init__(self, config):
        self.ini = config

    def quantize(self, model, dataloader_quant, f_keras="mct_quantized.keras"):
        self.setup_gpu()
        mct_model = self.run_mct(model, dataloader_quant)
        self.save(mct_model, f_keras)
        
    def setup_gpu(self, visible=-1):
        if visible is not None:
            tf_utility.cuda_visible(visible)
        print(
            "Num GPUs Available: ",
            len(tf_utility.physical_gpus()),
        )

    def run_mct(self, model, dataloader_quant_representative):
        # Quantize model using the representative dataset
        quantized_exportable_model, _ = mct.ptq.keras_post_training_quantization(
            model,
            dataloader_quant_representative,
        )
        print("Quantized model is ready")
        return quantized_exportable_model

    def save(self, quantized_exportable_model, f_mct_model, f_summary=None):
        print(f"saved quant model : {f_mct_model}")
        # Export a keras model with mctq custom quantizers.
        mct.exporter.keras_export_model(
            model=quantized_exportable_model, 
            save_model_path=f_mct_model,
        )
        if f_summary is not None:
            tf_utility.summary(quantized_exportable_model, f_summary)

import model_compression_toolkit as mct
import torch

from typing import Iterator, Tuple, List


class MctKeras:
    def __init__(self, config):
        # configuration
        self.config = config

    def _get_representative_dataset(
        self, n_iter: int, dataset_loader: Iterator[Tuple]
    ):
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
        quantized_model, quantization_info = (
            mct.ptq.keras_post_training_quantization(
                model,
                representative_data_gen=self._get_representative_dataset(
                    n_iter, dataloader
                ),
                target_platform_capabilities=tpc,
            )
        )

        print("Quantized model is ready")

        # export model
        target_path = self.config["PATH"]["QUANTIZED_KERAS"]
        print(f"saved {target_path}")
        quantized_model.save(target_path)

        return quantized_model

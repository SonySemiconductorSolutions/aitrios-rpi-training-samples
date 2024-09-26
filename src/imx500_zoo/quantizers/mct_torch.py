import model_compression_toolkit as mct

from model_compression_toolkit.exporter import PytorchExportSerializationFormat

import torch
import gc


class MctTorch:
    def __init__(self, config):
        # configuration
        self.config = config

    def quantize(self, model, dataloader):
        conf = self.config["QUANTIZER"]
        z_threshold = float(conf["Z_THRESHOLD"])
        calibration_iterations = int(conf["CALIB_ITERATIONS"])
        USE_DATASET_RAND = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        def _representative_data_gen() -> list:
            for _ in range(calibration_iterations):
                yield [next(iter(dataloader))[0]]

        for param in model.parameters():
            param.requires_grad = True

        # Perform post training quantization
        # set configuration and Configure z threshold algorithm for outlier removal.
        core_config = mct.core.CoreConfig(
            quantization_config=mct.core.QuantizationConfig(
                z_threshold=z_threshold
            )
        )
        tpc = mct.get_target_platform_capabilities("pytorch", "imx500")
        quantized_model = None

        quantized_model, quantization_info = (
            mct.ptq.pytorch_post_training_quantization(
                model,
                _representative_data_gen,
                core_config=core_config,
                target_platform_capabilities=tpc,
            )
        )
        print("Quantized model is ready")

        # clear GPU memory
        print("self.device = ", self.device)
        if self.device == "cuda":
            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()

        # export model
        target_path = self.config["PATH"]["QUANTIZED_ONNX"]

        mct.exporter.pytorch_export_model(
            model=quantized_model,
            save_model_path=target_path,
            repr_dataset=_representative_data_gen,
            serialization_format=PytorchExportSerializationFormat.ONNX,
        )

        return quantized_model

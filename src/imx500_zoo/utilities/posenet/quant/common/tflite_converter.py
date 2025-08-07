from __future__ import division
import tensorflow as tf
from .seg_mct_quantization import SegMCTQuantization
from model_compression_toolkit.logger import Logger
from tensorflow_model_optimization.python.core.quantization.keras import (
    quantize_config,
)

from mct_quantizers import (
    KerasActivationQuantizationHolder,
    KerasQuantizationWrapper,
)

from keras import backend as KB


logger = Logger.get_logger()


class NoOpQuantizeConfig(quantize_config.QuantizeConfig):
    """QuantizeConfig which does not quantize any part of the layer."""

    def get_weights_and_quantizers(self, layer):
        """dummy function"""
        return []

    def get_activations_and_quantizers(self, layer):
        """dummy function"""
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        """dummy function"""
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        """dummy function"""
        pass

    def get_output_quantizers(self, layer):
        """dummy function"""
        return []

    def get_config(self):
        """dummy function"""
        return {}


config = {"NoOpQuantizeConfig": NoOpQuantizeConfig, "K": KB}


class TFLiteConverter:
    """
    This class will be called for model conversion in different formats.
    """

    def __init__(self, config, input_size):
        self.config = config
        self.representative_data_path = self.config.model_config[
            "dataset_representative_data_path"
        ]
        self.mct_workflow_float32_float32_tflite_path = (
            self.config.mct_workflow_float32_float32_tflite_path
        )
        self.mct_workflow_mct_float32_tflite_mct_float32_tflite_path = (
            self.config.mct_workflow_mct_float32_tflite_mct_float32_tflite_path
        )
        self.mct_workflow_mct_int8_cpu_int8_mct_tflite_path = (
            self.config.mct_workflow_mct_int8_cpu_int8_mct_tflite_path
        )
        self.class_names = {0: "female", 1: "male"}
        self.gt_list = []
        self.pred_list = []
        self.input_size = input_size

    def keypoint_mct_convert(
        self, input_h5_model_path, output_model_name, output_model_name_keras=""
    ):
        model = tf.keras.models.load_model(input_h5_model_path, compile=False)
        input_size = self.input_size
        input_size = [int(input_size[0]), int(input_size[1])]
        mct_auantization_obj = SegMCTQuantization(
            self.config, model, self.representative_data_path, input_size
        )
        mct_convert_path = mct_auantization_obj.mct_quatize()
        model = tf.keras.models.load_model(
            mct_convert_path,
            custom_objects={
                "KerasActivationQuantizationHolder": KerasActivationQuantizationHolder,
                "KerasQuantizationWrapper": KerasQuantizationWrapper,
            },
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(output_model_name, "wb") as f:
            f.write(tflite_model)
        print("tflite mct conversion done")

        if not output_model_name_keras == "":
            model.save(output_model_name_keras)
            print(f"write quantized keras file : {output_model_name_keras}")


if __name__ == "__main__":
    tflite_converter = TFLiteConverter()
    tflite_converter.convert()

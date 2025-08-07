import cv2
import tensorflow as tf

# pip install model_compression_toolkit
import model_compression_toolkit as mct

from third_party.mct.data_loader import FolderImageLoader

from model_compression_toolkit.logger import Logger
from tensorflow_model_optimization.python.core.quantization.keras import (
    quantize_config,
)
import tensorflow_model_optimization as tfmot

from keras import backend as KB

logger = Logger.get_logger()


MEAN = 0
STD = 256.0


# MCT_QUANT_h5_MODEL_PATH = "/home/ubuntu/object_detection/tflite/mnasNet/weights_converted/multirace_age/samrai_MCT.h5"
def hard_swish(features):
    features = tf.convert_to_tensor(features)
    fdtype = features.dtype
    return features * tf.nn.relu6(features + tf.cast(3.0, fdtype)) * (1.0 / 6.0)


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


class SegMCTQuantization:
    def __init__(self, config, modelArchIn, repDataFldrIn, input_size):
        self.config = config
        self.modelArch = modelArchIn
        self.repDataFldr = repDataFldrIn
        self.input_size = input_size

        # output path

        self.mct_workflow_mct_float32_tflite_mct_h5_path = (
            self.config.mct_workflow_mct_float32_tflite_mct_h5_path
        )

        # self.SIZE = input_size[0]
        # self.SIZE_SCALE = 256 / self.SIZE

    def _resize(self, x):
        # resize_side = max(self.SIZE_SCALE * self.SIZE / x.shape[0], self.SIZE_SCALE * self.SIZE / x.shape[1])
        # height_tag = int(np.round(resize_side * x.shape[0]))
        # width_tag = int(np.round(resize_side * x.shape[1]))
        width = self.input_size[0]
        height = self.input_size[1]
        resized_img = cv2.resize(x, (height, width))
        # offset_height = int((height_tag - self.SIZE) / 2)
        # offset_width = int((width_tag - self.SIZE) / 2)
        # cropped_img = resized_img[offset_height:offset_height + self.SIZE, offset_width:offset_width + self.SIZE]
        # return cropped_img
        return resized_img

    def _normalization(self, x):
        return (x - MEAN) / STD
        # return x

    def _representative_data_gen(self):
        # Set the batch size of the images at each calibration iteration.
        batch_size = 1

        # Set the path to the folder of images to load and use for the representative dataset.
        # Notice that the folder have to contain at least one image.
        folder = self.repDataFldr

        # Create a representative data generator, which returns a list of images.
        # The images can be preprocessed using a list of preprocessing functions.
        image_data_loader = FolderImageLoader(
            folder,
            preprocessing=[self._resize, self._normalization],
            batch_size=batch_size,
        )

        # Create a Callable representative dataset for calibration purposes.
        # The function should be called without any arguments, and should return a list numpy arrays (array for each
        # model's input).
        # For example: A model has two input tensors - one with input shape of [32 X 32 X 3] and the second with
        # an input shape of [224 X 224 X 3]. We calibrate the model using batches of 20 images.
        # Calling representative_data_gen() should return a list
        # of two numpy.ndarray objects where the arrays' shapes are [(20, 3, 32, 32), (20, 3, 224, 224)].
        # def representative_data_gen():
        return [image_data_loader.sample()]

    def mct_quatize(self):
        # Get a TargetPlatformModel object that models the hardware for the quantized model inference.
        # The model determines the quantization methods to use during the MCT optimization process.
        # Here, for example, we use the default target platform model that is attached to a Tensorflow
        # layers representation.

        # Create a model and quantize it using the representative_data_gen as the calibration images.
        # Set the number of calibration iterations to 10.
        # MCT Quant
        custom_objects2 = {"hard_swish": hard_swish}
        with tfmot.quantization.keras.quantize_scope():
            with tf.keras.utils.custom_object_scope(custom_objects2):
                quantized_model, quantization_info = (
                    mct.ptq.keras.quantization_facade.keras_post_training_quantization(
                        #                    mct.keras_post_training_quantization(
                        self.modelArch,
                        self._representative_data_gen,
                        #                        n_iter=10,
                    )
                )
        logger.info(f"quantization_info = {quantization_info}")
        tf.keras.models.save_model(
            quantized_model, self.mct_workflow_mct_float32_tflite_mct_h5_path
        )
        return self.mct_workflow_mct_float32_tflite_mct_h5_path

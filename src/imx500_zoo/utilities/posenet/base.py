from imx500_zoo.utilities.posenet.config import Config

import imx500_zoo.trainers.posenet_trainer
import imx500_zoo.quantizers.mct_tflite
import imx500_zoo.utilities.posenet.quant.common.keypoint_callback
import imx500_zoo.utilities.posenet.quant.common.seg_mct_quantization
import imx500_zoo.validators.keras_posenet_validator
import imx500_zoo.utilities.posenet.data_generator
import imx500_zoo.utilities.posenet.utils.augmentation
import third_party.keraspersonlab.data_parser
import third_party.keraspersonlab.loss
import imx500_zoo.utilities.posenet.utils.metrics
import imx500_zoo.utilities.posenet.utils.model
import third_party.keraspersonlab.plot
import third_party.keraspersonlab.post_proc
import third_party.keraspersonlab.post_processing

from model_compression_toolkit.logger import Logger


def init_modules_base(config_i=None, fconfig=""):
    if config_i is None:
        config = Config(fconfig)
        config.parse_yaml()
    else:
        config = config_i

    init_quant_logger(config)

    imx500_zoo.trainers.posenet_trainer.set_config(config)
    imx500_zoo.quantizers.mct_tflite.set_config(config)
    imx500_zoo.validators.keras_posenet_validator.set_config(config)

    imx500_zoo.utilities.posenet.data_generator.set_config(config)
    imx500_zoo.utilities.posenet.utils.augmentation.set_config(config)
    third_party.keraspersonlab.data_parser.set_config(config)
    imx500_zoo.utilities.posenet.utils.metrics.set_config(config)
    imx500_zoo.utilities.posenet.utils.model.set_config(config)
    third_party.keraspersonlab.plot.set_config(config)
    third_party.keraspersonlab.post_proc.set_config(config)
    third_party.keraspersonlab.post_processing.set_config(config)
    third_party.keraspersonlab.loss.set_config(config)
    imx500_zoo.utilities.posenet.quant.common.keypoint_callback.set_config(config)


def init_quant_logger(config_i):
    logger = Logger.get_logger()
    Logger.set_log_file(config_i.LOGS_PATH)

    config_i.quant_logger = logger

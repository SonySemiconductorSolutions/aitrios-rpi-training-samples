from __future__ import division

# import mct_quant
from imx500_zoo.utilities.posenet.data_generator import (
    DataGenerator,
    DataGenType,
)
from imx500_zoo.utilities.posenet.quant.common.keypoint_callback import (
    Keypoint_callback_inference,
)
from model_compression_toolkit.logger import Logger

logger = Logger.get_logger()


class TFliteInference:
    def __init__(self, config):
        self.gt_list = []
        self.pred_list = []
        self.config = config
        self.class_names = self._get_classnames()
        logger.info("self.class_names", self.class_names)

    def _get_classnames(self):
        return self.config.config_json.get_classnames()

    def keypoint_inference(
        self,
        model,
        testDataFldr,
        annotation_file,
        input_size,
        nb_kp,
        dataloader=None,
    ):
        logs = {}
        if model[-2:] != "h5":
            val_gen_return_kp = (
                DataGenerator(
                    phase="test",
                    BATCH_SIZE=4,
                    train_val_split=0.8,
                    seed=0.5,
                    return_kp=True,
                    testDataFldr=testDataFldr,
                    annotation_file=annotation_file,
                    input_size=input_size,
                    nb_kp=nb_kp,
                    dg_type=DataGenType.QUANT,
                )
                if dataloader is None
                else dataloader
            )
            val_steps = len(val_gen_return_kp)

            pck_compute = Keypoint_callback_inference(
                val_gen_return_kp,
                "PCK_FULL",
                model,
                val_steps,
                False,
                mode="AP",
            )
            pck_compute.on_epoch_end()
            pck_compute.metric_value
            logs["pck AP"] = pck_compute.get_metric()

            pck_compute = Keypoint_callback_inference(
                val_gen_return_kp,
                "PCK_FULL",
                model,
                val_steps,
                False,
                mode="AR",
            )
            pck_compute.on_epoch_end()
            pck_compute.metric_value
            logs["pck AR"] = pck_compute.get_metric()

            OKS_FULL_compute = Keypoint_callback_inference(
                val_gen_return_kp,
                "OKS_FULL",
                model,
                val_steps,
                False,
                mode="AP",
            )
            OKS_FULL_compute.on_epoch_end()
            OKS_FULL_compute.metric_value
            logs["oks AP"] = OKS_FULL_compute.get_metric()

            OKS_FULL_compute = Keypoint_callback_inference(
                val_gen_return_kp,
                "OKS_FULL",
                model,
                val_steps,
                False,
                mode="AR",
            )
            OKS_FULL_compute.on_epoch_end()
            OKS_FULL_compute.metric_value
            logs["oks AR"] = OKS_FULL_compute.get_metric()
        else:
            val_gen_return_kp = (
                DataGenerator(
                    phase="test",
                    BATCH_SIZE=1,
                    train_val_split=0.8,
                    seed=0.5,
                    return_kp=True,
                    testDataFldr=testDataFldr,
                    annotation_file=annotation_file,
                    input_size=input_size,
                    nb_kp=nb_kp,
                    dg_type=DataGenType.QUANT,
                )
                if dataloader is None
                else dataloader
            )
            val_steps = len(val_gen_return_kp)

            pck_compute = Keypoint_callback_inference(
                val_gen_return_kp,
                "PCK_FULL",
                model,
                val_steps,
                False,
                mode="AP",
            )
            pck_compute.on_epoch_end_h5()
            pck_compute.metric_value
            logs["pck AP"] = pck_compute.get_metric()

            pck_compute = Keypoint_callback_inference(
                val_gen_return_kp,
                "PCK_FULL",
                model,
                val_steps,
                False,
                mode="AR",
            )
            pck_compute.on_epoch_end_h5()
            pck_compute.metric_value
            logs["pck AR"] = pck_compute.get_metric()

            OKS_FULL_compute = Keypoint_callback_inference(
                val_gen_return_kp,
                "OKS_FULL",
                model,
                val_steps,
                False,
                mode="AP",
            )
            OKS_FULL_compute.on_epoch_end_h5()
            OKS_FULL_compute.metric_value
            logs["oks AP"] = OKS_FULL_compute.get_metric()

            OKS_FULL_compute = Keypoint_callback_inference(
                val_gen_return_kp,
                "OKS_FULL",
                model,
                val_steps,
                False,
                mode="AR",
            )
            OKS_FULL_compute.on_epoch_end_h5()
            OKS_FULL_compute.metric_value
            logs["oks AR"] = OKS_FULL_compute.get_metric()

        return logs

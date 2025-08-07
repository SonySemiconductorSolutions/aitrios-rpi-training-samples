import tensorflow as tf
from imx500_zoo.utilities.posenet.utils.model import get_personlab_model
from third_party.tensorflow.mobilenet import get_mobilenetv1_base
from third_party.keraspersonlab.post_processing import Post_Process
from imx500_zoo.utilities.posenet.data_generator import DataGenerator
import numpy as np
import time

from imx500_zoo.utilities.posenet.utils.metrics import (
    OKS,
    PCK,
    OKS_FULL,
    PCK_FULL,
)
from tqdm import tqdm

BATCH_SIZE = 1

config = None


def set_config(config_i):
    global config
    config = config_i


class Keypoint_callback_inference:
    """
    Class for OKS/PCK Metric computation callback.
    """

    def __init__(self, validation, metric="OKS", model=None, val_steps=None, mode="AR"):
        self.validation = validation
        self.metric = metric
        self.val_steps = val_steps

        if metric == "OKS":
            self._metric = OKS()
        elif metric == "PCK":
            self._metric = PCK()
        elif metric == "OKS_FULL":
            self._metric = OKS_FULL(mode)
        elif metric == "PCK_FULL":
            self._metric = PCK_FULL(mode)

        self.model = model
        self.post_proc = Post_Process()

    def on_epoch_end(self, epoch=1, logs={}):
        st_time = time.time()
        self._metric.reset_state()

        logs["{}_val".format(self.metric)] = float("inf")
        count = 0
        metric_value_list = []

        for images, labels, keypoint_dict in tqdm(self.validation):
            outputs_kpmaps, outputs_shortoffset, outputs_midoffset = self.model.predict(
                images
            )
            self.post_proc.reset()
            outputs_kpmaps = self.post_proc.tensor_post_processing(outputs_kpmaps, 1)
            outputs_shortoffset = self.post_proc.tensor_post_processing(
                outputs_shortoffset, 2
            )

            if (self.metric == "OKS_FULL") or (self.metric == "PCK_FULL"):
                outputs_midoffset = self.post_proc.tensor_post_processing(
                    outputs_midoffset, 3
                )
                self.post_proc.reset()
                self._metric.update_state(
                    keypoint_dict,
                    outputs_kpmaps,
                    outputs_shortoffset,
                    outputs_midoffset,
                )
            else:
                self.post_proc.reset()
                self._metric.update_state(
                    keypoint_dict, outputs_kpmaps, outputs_shortoffset
                )

            metric_value_list.append(self._metric.result())
            self._metric.reset_state()
            count += 1
            if count == self.val_steps:
                break

        metric_value_tensor = tf.stack(metric_value_list)
        self.metric_value = tf.reduce_mean(metric_value_tensor)

        logs["{}_val".format(self.metric)] = np.round(self.metric_value.numpy(), 3)

        en_time = time.time()
        print(
            "Elapsed time for {} calculation : ".format(self.metric),
            en_time - st_time,
        )

    def get_metric(metric_value):
        return np.round(float(metric_value.numpy()), 3)


class KerasPosenetValidator:
    def __init__(self, config):
        self.config = config
        self.dataloader_eval = None

    def validate(self, dataloader_eval):
        self.dataloader_eval = dataloader_eval
        self.run()
        return None

    def run(self):
        # #select GPU explicitly for different experiments
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.set_visible_devices(gpus[config.GPU_ID], "GPU")
            except Exception as e:
                print(f"GPU unavailable : {e}")

        model_path = config.TRAIN_MODEL_H5
        model = get_personlab_model(get_mobilenetv1_base)
        model.load_weights(model_path)

        results = []

        # PCK callback metric
        val_gen_return_kp = (
            DataGenerator(BATCH_SIZE, "test", 0.8, seed=0.5, return_kp=True)
            if self.dataloader_eval is None
            else self.dataloader_eval
        )
        val_steps = len(val_gen_return_kp)
        pck_compute = Keypoint_callback_inference(
            val_gen_return_kp, "PCK_FULL", model, val_steps, mode="AR"
        )
        pck_compute.on_epoch_end()
        pck_compute.metric_value
        print("pck AR :", pck_compute.metric_value)
        results.append(pck_compute.metric_value)

        # OKS callback metric
        #        from imx500_zoo.utilities.posenet.utils.metrics import *

        OKS_FULL_compute = Keypoint_callback_inference(
            val_gen_return_kp, "OKS_FULL", model, val_steps, mode="AR"
        )
        OKS_FULL_compute.on_epoch_end()
        print("oks AR :", OKS_FULL_compute.metric_value)
        results.append(OKS_FULL_compute.metric_value)

        # OKS callback metric
        #        from imx500_zoo.utilities.posenet.utils.metrics import *

        OKS_FULL_compute = Keypoint_callback_inference(
            val_gen_return_kp, "OKS_FULL", model, val_steps, mode="AP"
        )
        OKS_FULL_compute.on_epoch_end()
        print("oks AP :", OKS_FULL_compute.metric_value)
        results.append(OKS_FULL_compute.metric_value)

        #        from imx500_zoo.utilities.posenet.utils.metrics import *

        pck_compute = Keypoint_callback_inference(
            val_gen_return_kp, "PCK_FULL", model, val_steps, mode="AP"
        )
        pck_compute.on_epoch_end()
        pck_compute.metric_value
        print("pck AP :", pck_compute.metric_value)
        results.append(pck_compute.metric_value)

        print("  -- summary --")
        print(f"model : {config.EVALUATE_MODEL_PATH}")
        print("pck AR :", results[0])
        print("oks AR :", results[1])
        print("oks AP :", results[2])
        print("pck AP :", results[3])

        dumps = self.config.results.valid
        dumps["MODEL"] = model_path
        dumps["pck AR"] = self.get_result(results[0])
        dumps["pck AP"] = self.get_result(results[3])
        dumps["oks AR"] = self.get_result(results[1])
        dumps["oks AP"] = self.get_result(results[2])

    def get_result(self, metric_value):
        return Keypoint_callback_inference.get_metric(metric_value)

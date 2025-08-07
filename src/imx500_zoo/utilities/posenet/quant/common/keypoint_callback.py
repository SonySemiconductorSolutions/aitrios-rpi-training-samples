from __future__ import print_function
import os
import time
import multiprocessing
import tensorflow as tf
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
from third_party.keraspersonlab.post_processing import Post_Process
from model_compression_toolkit.logger import Logger
from imx500_zoo.utilities.posenet.utils.metrics import (
    OKS,
    PCK,
    OKS_FULL,
    PCK_FULL,
)

logger = Logger.get_logger()
workers = multiprocessing.cpu_count() // 2
config = None


def set_config(config_i):
    global config

    config = config_i


class Keypoint_callback_inference:
    """
    Class for OKS/PCK Metric computation callback.
    """

    def __init__(
        self,
        validation,
        metric="OKS",
        model=None,
        val_steps=None,
        emet=False,
        emet_parse_out_list=None,
        testDataFldr=None,
        input_size=None,
        mode="AR",
    ):
        self.validation = validation
        self.metric = metric
        self.val_steps = val_steps
        self.emet = emet
        self.emet_parse_out_list = emet_parse_out_list
        self.testDataFldr = testDataFldr
        self.input_size = input_size
        self.mode = mode
        if metric == "OKS":
            self._metric = OKS()
        elif metric == "PCK":
            self._metric = PCK()
        elif metric == "OKS_FULL":
            self._metric = OKS_FULL(mode)
        elif metric == "PCK_FULL":
            self._metric = PCK_FULL(mode)
        elif metric == "BinaryAccuracy":
            self._metric = tf.keras.metrics.BinaryAccuracy()
        self.model = model
        self.post_proc = Post_Process()

    def multiprocessing_inference_keypoints(self, func, x_img, workers):
        with ProcessPoolExecutor(workers) as ex:
            res = ex.map(func, x_img)
        return list(res)

    def read_from_opencvMat_keypoints_parallel(self, img):
        image = np.expand_dims(img, axis=0)
        model = tf.lite.Interpreter(model_path=self.model)
        model.allocate_tensors()
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]["index"], image)
        model.invoke()
        # output_data = model.get_tensor(output_details[1]['index'])
        d = {}
        for i in range(len(output_details)):
            d[str(model.get_tensor(output_details[i]["index"]).shape[-1])] = (
                model.get_tensor(output_details[i]["index"])
            )
        outputs_kpmaps = d[str(config.NUM_KP)]
        outputs_shortoffset = d[str(config.NUM_KP * 2)]
        outputs_midoffset = d[str(config.NUM_EDGES * 4)]
        return outputs_kpmaps, outputs_shortoffset, outputs_midoffset

    def get_image_path(self, bin_file_name):
        # TO DO from yaml label path
        label_path = self.testDataFldr
        regex_name = os.path.split(bin_file_name)[1][:-8]
        image_path_list = [x for x in os.listdir(label_path) if regex_name in x]
        return image_path_list[0]

    def on_epoch_end(self, epoch=1, logs={}):
        st_time = time.time()
        self._metric.reset_state()

        logs["{}_val".format(self.metric)] = float("inf")
        count = 0
        metric_value_list = []

        for images, labels, keypoint_dict in self.validation:
            res = self.multiprocessing_inference_keypoints(
                self.read_from_opencvMat_keypoints_parallel, images, workers
            )
            count += 1

            for r in range(len(res)):
                outputs_kpmaps = res[r][0]
                outputs_shortoffset = res[r][1]
                outputs_midoffset = res[r][2]
                self.post_proc.reset()
                outputs_kpmaps = self.post_proc.tensor_post_processing(
                    outputs_kpmaps, 1
                )
                outputs_shortoffset = self.post_proc.tensor_post_processing(
                    outputs_shortoffset, 2
                )

                keypoint_dict_new = {}
                keypoint_dict_new[str(0)] = keypoint_dict[list(keypoint_dict.keys())[r]]
                if (self.metric == "OKS_FULL") or (self.metric == "PCK_FULL"):
                    outputs_midoffset = self.post_proc.tensor_post_processing(
                        outputs_midoffset, 3
                    )
                    self.post_proc.reset()
                    self._metric.update_state(
                        keypoint_dict_new,
                        outputs_kpmaps,
                        outputs_shortoffset,
                        outputs_midoffset,
                    )
                elif self.metric == "BinaryAccuracy":
                    self.post_proc.reset()
                    self._metric.update_state(labels[0], outputs_kpmaps)
                else:
                    self.post_proc.reset()
                    self._metric.update_state(
                        keypoint_dict_new, outputs_kpmaps, outputs_shortoffset
                    )

                metric_value_list.append(self._metric.result())
                self._metric.reset_state()
            if count == self.val_steps:
                break

        metric_value_tensor = tf.stack(metric_value_list)
        self.metric_value = tf.reduce_mean(metric_value_tensor)

        logger.info("metric_value = " + str(np.round(self.metric_value.numpy(), 3)))
        en_time = time.time()
        print(
            "Elapsed time for {} {} calculation : ".format(self.metric, self.mode),
            en_time - st_time,
        )

    def on_epoch_end_h5(self, epoch=1, logs={}):
        st_time = time.time()
        self._metric.reset_state()

        logs["{}_val".format(self.metric)] = float("inf")
        count = 0
        metric_value_list = []

        for images, labels, keypoint_dict in self.validation:
            model = tf.keras.models.load_model(self.model, compile=False)
            outputs_kpmaps, outputs_shortoffset, outputs_midoffset = model.predict(
                images
            )
            self.post_proc.reset()
            outputs_kpmaps = self.post_proc.tensor_post_processing(outputs_kpmaps, 1)
            outputs_shortoffset = self.post_proc.tensor_post_processing(
                outputs_shortoffset, 2
            )

            keypoint_dict_new = {}
            for i, key in enumerate(keypoint_dict.keys()):
                keypoint_dict_new[str(i)] = keypoint_dict[key]

            if (self.metric == "OKS_FULL") or (self.metric == "PCK_FULL"):
                outputs_midoffset = self.post_proc.tensor_post_processing(
                    outputs_midoffset, 3
                )
                self.post_proc.reset()
                self._metric.update_state(
                    keypoint_dict_new,
                    outputs_kpmaps,
                    outputs_shortoffset,
                    outputs_midoffset,
                )
            elif self.metric == "BinaryAccuracy":
                self.post_proc.reset()
                self._metric.update_state(labels[0], outputs_kpmaps)
            else:
                self.post_proc.reset()
                self._metric.update_state(
                    keypoint_dict_new, outputs_kpmaps, outputs_shortoffset
                )

            metric_value_list.append(self._metric.result())
            self._metric.reset_state()
            count += 1
            if count == self.val_steps:
                break

        metric_value_tensor = tf.stack(metric_value_list)
        self.metric_value = tf.reduce_mean(metric_value_tensor)

        logger.info("metric_value = " + str(np.round(self.metric_value.numpy(), 3)))
        en_time = time.time()
        print(
            "Elapsed time for {} {} calculation : ".format(self.metric, self.mode),
            en_time - st_time,
        )

    def get_metric(self):
        return np.round(float(self.metric_value.numpy()), 3)

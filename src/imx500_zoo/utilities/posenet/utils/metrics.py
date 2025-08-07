import tensorflow as tf
import numpy as np
from keras import backend as KB
from third_party.keraspersonlab.post_processing import Post_Process
from third_party.keraspersonlab.post_proc import (
    get_keypoints_pck_metrics,
    get_keypoints_pck_metrics_full,
    get_keypoints_oks_metrics,
    get_keypoints_oks_metrics_full,
    compute_heatmaps,
    gaussian_filter,
)

import time

input_shape = None
class_names = None
post_proc = None

config = None


def set_config(config_i):
    global config
    global input_shape
    global class_names

    config = config_i
    input_shape = [config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]]
    class_names = config.KEYPOINTS


def identity_metric(y_true, y_pred):
    return KB.mean(y_pred)


class BinaryAccuracy(tf.keras.metrics.Metric):
    """
    Class for TopKAccuracy Metric computation.
    """

    def __init__(self) -> None:
        """

        It will be executed just once before any accumulate or compute
        is being called.
        """
        super().__init__()
        self._metric = tf.keras.metrics.BinaryAccuracy()

        global post_proc
        if post_proc is None:
            post_proc = Post_Process()

    def update_state(
        self, targets: tf.Tensor, outputs: tf.Tensor, sample_weight=None
    ) -> None:
        """Accumulate values from the outputs and annotations. It comes in a
        batch format.

        :param outputs: The output Tensor from the custom model.

        :param targets: The targets Tensor generated when preprocessing.

        """
        targets = targets[:, :, :, 0 : config.NUM_KP]
        outputs = post_proc.tensor_post_processing(outputs, 1)
        post_proc.previous_state()
        post_proc.extra_state(targets)
        post_proc.reset()

        self._metric.update_state(targets, outputs)

    def result(self) -> tf.Tensor:
        """Compute the metric given the accumulated values."""
        return self._metric.result()

    def reset_state(self) -> None:
        """Reset the accumulated values."""
        self._metric.reset_state()


class Keypoint_PCK(tf.keras.metrics.Metric):
    """
    Class for PCK Metric computation.
    """

    def __init__(self) -> None:
        """

        It will be executed just once before any accumulate or compute
        is being called.
        """
        super().__init__()
        self._metric = PCK()

        global post_proc
        if post_proc is None:
            post_proc = Post_Process()

    def update_state(
        self, targets: tf.Tensor, outputs: tf.Tensor, sample_weight=None
    ) -> None:
        """Accumulate values from the outputs and annotations. It comes in a
        batch format.

        :param outputs: The output Tensor from the custom model.

        :param targets: The targets Tensor generated when preprocessing.

        """
        post_proc.reset()
        outputs_kpmaps = post_proc.kp_maps_prev

        outputs_shortoffset = post_proc.tensor_post_processing(outputs, 2)
        post_proc.reset()

        targets_kpmaps = post_proc.extra_input
        targets_shortoffset = targets

        self._metric.update_state(
            targets_kpmaps,
            targets_shortoffset,
            outputs_kpmaps,
            outputs_shortoffset,
        )

    def result(self) -> tf.Tensor:
        """Compute the metric given the accumulated values."""
        return self._metric.result()

    def reset_state(self) -> None:
        """Reset the accumulated values."""
        self._metric.reset_state()


class PCK:
    def __init__(self):
        self.PCK = 0.0

    def compute_pck(self, keypoint_dict, outputs1, outputs2):
        if isinstance(outputs1, np.ndarray):
            batch_size = outputs1.shape[0]
        else:
            batch_size = outputs1.numpy().shape[0]
        total_AP = []

        for i in range(batch_size):
            if isinstance(outputs1, np.ndarray) or isinstance(outputs2, np.ndarray):
                output_kp_maps = outputs1[i]
                output_short_offset = outputs2[i]
            else:
                output_kp_maps = outputs1.numpy()[i]
                output_short_offset = outputs2.numpy()[i]

            outputs = [output_kp_maps, output_short_offset]

            AP_dict = {class_name: 0.0 for class_name in class_names}

            H_pred = compute_heatmaps(kp_maps=outputs[0], short_offsets=outputs[1])

            # Gaussian filtering helps when there are multiple local maxima for the same keypoint.
            for q in range(config.NUM_KP):
                H_pred[:, :, q] = gaussian_filter(H_pred[:, :, q], sigma=2)

            summary_dict = get_keypoints_pck_metrics(keypoint_dict[str(i)], H_pred)

            # calculate AP for each class
            for w, class_name in enumerate(class_names):
                if summary_dict[str(w)][0] == 0 and summary_dict[str(w)][1] == 0:
                    AP_dict[class_name] = 0
                else:
                    AP_dict[class_name] = (
                        summary_dict[str(w)][0]
                        * 1.0
                        / (summary_dict[str(w)][0] + summary_dict[str(w)][1])
                    )

            # get AP
            total_AP.append(list(AP_dict.values()))

        return np.sum(total_AP) / (config.NUM_KP * batch_size)

    def update_state(
        self, keypoint_dict, outputs1: tf.Tensor, outputs2: tf.Tensor
    ) -> None:
        self.PCK = self.compute_pck(keypoint_dict, outputs1, outputs2)

    def result(self) -> tf.Tensor:
        """Compute the metric given the accumulated values."""
        return self.PCK

    def reset_state(self) -> None:
        """Reset the accumulated values."""
        self.PCK = 0.0


class OKS:
    def __init__(self):
        self.OKS = 0.0

    def compute_oks(self, keypoint_dict, outputs1, outputs2):
        if isinstance(outputs1, np.ndarray):
            batch_size = outputs1.shape[0]
        else:
            batch_size = outputs1.numpy().shape[0]
        total_AP = []

        for i in range(batch_size):
            if isinstance(outputs1, np.ndarray) or isinstance(outputs2, np.ndarray):
                output_kp_maps = outputs1[i]
                output_short_offset = outputs2[i]
            else:
                output_kp_maps = outputs1.numpy()[i]
                output_short_offset = outputs2.numpy()[i]

            outputs = [output_kp_maps, output_short_offset]

            AP_dict = {class_name: 0.0 for class_name in class_names}

            H_pred = compute_heatmaps(kp_maps=outputs[0], short_offsets=outputs[1])

            # Gaussian filtering helps when there are multiple local maxima for the same keypoint.
            for q in range(config.NUM_KP):
                H_pred[:, :, q] = gaussian_filter(H_pred[:, :, q], sigma=2)

            summary_dict = get_keypoints_oks_metrics(keypoint_dict[str(i)], H_pred)

            # calculate AP for each class
            for w, class_name in enumerate(class_names):
                if summary_dict[str(w)][0] == 0 and summary_dict[str(w)][1] == 0:
                    AP_dict[class_name] = 0
                else:
                    AP_dict[class_name] = (
                        summary_dict[str(w)][0]
                        * 1.0
                        / (summary_dict[str(w)][0] + summary_dict[str(w)][1])
                    )

            # get AP
            total_AP.append(list(AP_dict.values()))

        return np.sum(total_AP) / (config.NUM_KP * batch_size)

    def update_state(
        self, keypoint_dict, outputs1: tf.Tensor, outputs2: tf.Tensor
    ) -> None:
        self.OKS = self.compute_oks(keypoint_dict, outputs1, outputs2)

    def result(self) -> tf.Tensor:
        """Compute the metric given the accumulated values."""
        return self.OKS

    def reset_state(self) -> None:
        """Reset the accumulated values."""
        self.OKS = 0.0


class OKS_FULL:
    # compute oks for entire keypoints instead of a single class
    def __init__(self, metric="AR"):
        self.OKS_FULL = 0.0
        self.metric = metric

    def compute_oks(self, keypoint_dict, outputs1, outputs2, outputs3):
        if isinstance(outputs1, np.ndarray):
            batch_size = outputs1.shape[0]
        else:
            batch_size = outputs1.numpy().shape[0]
        total_tps = []
        total_fps = []
        total_fns = []
        for i in range(batch_size):
            if isinstance(outputs1, np.ndarray) or isinstance(outputs2, np.ndarray):
                output_kp_maps = outputs1[i]
                output_short_offset = outputs2[i]
                output_mid_offset = outputs3[i]
            else:
                output_kp_maps = outputs1.numpy()[i]
                output_short_offset = outputs2.numpy()[i]
                output_mid_offset = outputs3.numpy()[i]

            outputs = [output_kp_maps, output_short_offset, output_mid_offset]

            H_pred = compute_heatmaps(kp_maps=outputs[0], short_offsets=outputs[1])

            # Gaussian filtering helps when there are multiple local maxima for the same keypoint.
            for q in range(config.NUM_KP):
                H_pred[:, :, q] = gaussian_filter(H_pred[:, :, q], sigma=2)

            TP_per_image, FP_per_image, FN_per_image = get_keypoints_oks_metrics_full(
                keypoint_dict[str(i)], H_pred, outputs[2]
            )

            if TP_per_image * FP_per_image * FN_per_image > -1:
                total_tps.append(TP_per_image)
                total_fps.append(FP_per_image)
                total_fns.append(FN_per_image)

        if self.metric == "AP":
            if np.sum(total_tps) == 0 and np.sum(total_fps) == 0:
                return 0
            else:
                return np.sum(total_tps) / (np.sum(total_tps) + np.sum(total_fps))

        if self.metric == "AR":
            if np.sum(total_tps) == 0 and np.sum(total_fns) == 0:
                return 0
            else:
                return np.sum(total_tps) / (np.sum(total_tps) + np.sum(total_fns))

    def update_state(
        self, keypoint_dict, outputs1: tf.Tensor, outputs2: tf.Tensor, outputs3
    ) -> None:
        self.OKS_FULL = self.compute_oks(keypoint_dict, outputs1, outputs2, outputs3)

    def result(self) -> tf.Tensor:
        """Compute the metric given the accumulated values."""
        return self.OKS_FULL

    def reset_state(self) -> None:
        """Reset the accumulated values."""
        self.OKS_FULL = 0.0


class PCK_FULL:
    # compute pck for entire keypoints instead of a single class
    def __init__(self, metric="AR"):
        self.PCK_FULL = 0.0
        self.metric = metric

    def compute_pck(self, keypoint_dict, outputs1, outputs2, outputs3):
        if isinstance(outputs1, np.ndarray):
            batch_size = outputs1.shape[0]
        else:
            batch_size = outputs1.numpy().shape[0]

        total_tps = []
        total_fps = []
        total_fns = []
        for i in range(batch_size):
            if isinstance(outputs1, np.ndarray) or isinstance(outputs2, np.ndarray):
                output_kp_maps = outputs1[i]
                output_short_offset = outputs2[i]
                output_mid_offset = outputs3[i]
            else:
                output_kp_maps = outputs1.numpy()[i]
                output_short_offset = outputs2.numpy()[i]
                output_mid_offset = outputs3.numpy()[i]

            outputs = [output_kp_maps, output_short_offset, output_mid_offset]

            H_pred = compute_heatmaps(kp_maps=outputs[0], short_offsets=outputs[1])

            # Gaussian filtering helps when there are multiple local maxima for the same keypoint.
            for q in range(config.NUM_KP):
                H_pred[:, :, q] = gaussian_filter(H_pred[:, :, q], sigma=2)

            TP_per_image, FP_per_image, FN_per_image = get_keypoints_pck_metrics_full(
                keypoint_dict[str(i)], H_pred, outputs[2]
            )

            if TP_per_image * FP_per_image * FN_per_image > -1:
                total_tps.append(TP_per_image)
                total_fps.append(FP_per_image)
                total_fns.append(FN_per_image)

        if self.metric == "AP":
            if np.sum(total_tps) == 0 and np.sum(total_fps) == 0:
                return 0
            else:
                return np.sum(total_tps) / (np.sum(total_tps) + np.sum(total_fps))

        if self.metric == "AR":
            if np.sum(total_tps) == 0 and np.sum(total_fns) == 0:
                return 0
            else:
                return np.sum(total_tps) / (np.sum(total_tps) + np.sum(total_fns))

    def update_state(
        self, keypoint_dict, outputs1: tf.Tensor, outputs2: tf.Tensor, outputs3
    ) -> None:
        self.PCK_FULL = self.compute_pck(keypoint_dict, outputs1, outputs2, outputs3)

    def result(self) -> tf.Tensor:
        """Compute the metric given the accumulated values."""
        return self.PCK_FULL

    def reset_state(self) -> None:
        """Reset the accumulated values."""
        self.PCK_FULL = 0.0


class Keypoint_callback(tf.keras.callbacks.Callback):
    """
    Class for OKS/PCK Metric computation callback.
    """

    def __init__(
        self,
        train=None,
        validation=None,
        train_steps=None,
        val_steps=None,
        metric="OKS",
    ):
        super(Keypoint_callback, self).__init__()
        self.train = train
        self.validation = validation
        self.metric = metric
        if metric == "OKS":
            self._metric = OKS()
        elif metric == "PCK":
            self._metric = PCK()
        elif metric == "OKS_FULL":
            self._metric = OKS_FULL()
        elif metric == "PCK_FULL":
            self._metric = PCK_FULL()

        self.train_steps, self.val_steps = train_steps, val_steps
        self.post_proc = Post_Process()

    def on_epoch_end(self, epoch, logs={}):
        st_time = time.time()
        self._metric.reset_state()

        if self.train:
            logs["{}_train".format(self.metric)] = float("inf")
            count = 0
            metric_value_list = []

            for images, labels, keypoint_dict in self.train():
                outputs_kpmaps, outputs_shortoffset, outputs_midoffset = (
                    self.model.predict(images)
                )
                self.post_proc.reset()
                outputs_kpmaps = self.post_proc.tensor_post_processing(
                    outputs_kpmaps, 1
                )
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
                if count == self.train_steps:
                    break
            metric_value_tensor = tf.stack(metric_value_list)
            metric_value = tf.reduce_mean(metric_value_tensor)
            logs["{}_train".format(self.metric)] = np.round(metric_value.numpy(), 3)

        if self.validation:
            logs["{}_val".format(self.metric)] = float("inf")
            count = 0
            metric_value_list = []

            for images, labels, keypoint_dict in self.validation:
                outputs_kpmaps, outputs_shortoffset, outputs_midoffset = (
                    self.model.predict(images)
                )
                self.post_proc.reset()
                outputs_kpmaps = self.post_proc.tensor_post_processing(
                    outputs_kpmaps, 1
                )
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

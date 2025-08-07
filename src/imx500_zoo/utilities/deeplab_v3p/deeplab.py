import tensorflow as tf
import keras.backend as KerasBackend

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import yaml
import os

from imx500_zoo.utilities.misc import EmptyClass
from imx500_zoo.utilities import conv

class DeepLab:
    def parse_config(self, param):
        """
        read yml file data and parse
        """
        config = EmptyClass()

        config.param_yml = param
        m = param.get("model_param")
        t = param.get("train_param")
        config.data_rootpath = param.get("data_rootpath")
        self._set_image(config, width=m.get("image_width"), height=m.get("image_height"))

        config.model_name = param.get("model_name")
        config.load_pretrained_weights = param.get("load_weights", 0)
        config.backbone = param.get("backbone")
        config.backbone_pretrained_weights = param.get("backbone_pretrained_weights", "None")
        if config.backbone_pretrained_weights == "None":
            config.backbone_pretrained_weights = None
        config.n_classes = m.get("num_class")
        config.classes = m.get("classes")
        config.batch_size = t.get("batch_size")
        config.epochs = t.get("num_epochs")
        config.learn_rate = t.get("learn_rate")
        config.gpu_id = t.get("gpu_id")
        config.rep_dataset = param.get("rep_dataset")
        config.best_weights = param.get("best_weights")
        config.mct_model = param.get("mct_model")
        config.conf_matrix = param.get("conf_matrix")
        config.weights_dir = param.get("weights_dir")
        config.log_dir = param.get("log_dir")

        self.config = config
        return config
    
    def update_ini(self, config, ini):
        m = ini["MODEL"]
        t = ini["TRAINER"]
        p = ini["PATH"]

        config.SOLUTION_NAME = ini["SOLUTION"]["NAME"]

        config.n_classes = int(m["NUM_CLASSES"])
        config.n_classes_wo_void = config.n_classes - 1

        isize = conv.list(m["INPUT_SIZE"])
        self._set_image(config, width=isize[0], height=isize[1])

        config.batch_size = int(t["BATCH_SIZE"])
        config.epochs = int(t["NUM_EPOCHS"])
        config.learn_rate = float(t["LEARNING_RATE"])

        config.PATH_MODEL = p["MODEL"]
        config.PATH_DATA = p["DATA"]
        config.PATH_LOG = p["LOG"]
        config.PATH_KERAS = p["KERAS"]
        config.PATH_KERAS_SUMMARY = p["KERAS_SUMMARY"]
        config.PATH_QUANTIZED_KERAS = p["QUANTIZED_KERAS"]
        config.PATH_QUANTIZED_KERAS_SUMMARY = p["QUANTIZED_KERAS_SUMMARY"]

        config.log_dir = os.path.join(config.PATH_LOG, config.SOLUTION_NAME)
        config.weights_dir = config.PATH_MODEL
        config.best_weights = config.PATH_KERAS
        config.mct_model = config.PATH_QUANTIZED_KERAS
        config.conf_matrix = os.path.join(config.PATH_MODEL, "confusion_matrix.png")
        config.rep_dataset = os.path.join(config.data_rootpath, "train/JPEGimages/")

        os.makedirs(config.log_dir, exist_ok=True)

        return config

    def _set_image(self, config, width, height):
        config.image_width = width
        config.image_height = height
        config.image_size = (
            config.image_width,
            config.image_height,
        )

    def read_yaml(self, f_yaml):
        config_file = open(f_yaml, "r")
        param = yaml.safe_load(config_file)
        return param

    def miou(self, targets, outputs):
        """
        Compute the IoU(Intersection over Union) for segmentation tasks.

        This function calculates the mean IoU for each class, excluding
        the first (background) and last (void) labels during training.

        Parameters:
        -----------
        targets : tensor
            Target as ground truth labels, with shape (batch_size, height, width, 1).
        outputs : tensor
            output as predicted labels, with shape (batch_size, height, width, num_classes).

        Returns:
        --------
        tensor
            The mean IoU across all classes, excluding the first and
            last labels.
        """
        outputs = self._reshape_outputs(outputs)
        metrics = self._ious(targets, self._top(outputs))
        return KerasBackend.mean(metrics[(~tf.math.is_nan(metrics))])

    def accuracy(self, targets, outputs):
        """
        Compute sparse accuracy, ignoring the last label.

        This function calculates the accuracy of predictions for segmentation
        tasks, while ignoring the last label.

        Parameters:
        -----------
        ref : tensor
            Ground truth labels, with shape (batch_size, height, width, 1).
        imp : tensor
            Predicted labels, with shape (batch_size, height, width, num_classes).

        Returns:
        --------
        tensor
            The computed accuracy value.
        """
        tars, valids, targets = self._precision_targets(targets)
        outs = self._precision_outputs(targets, outputs, valids)
        return outs / tars

    def categorical_crossentropy(self, targets, outputs):
        """
        Compute sparse categorical crossentropy loss, ignoring the last label.

        This function calculates the sparse categorical crossentropy loss for
        segmentation tasks, while ignoring the last label.

        Parameters:
        -----------
        ref : tensor
            Ground truth labels, with shape (batch_size, height, width, 1).
        imp : tensor
            Predicted labels, with shape (batch_size, height, width, num_classes).

        Returns:
        --------
        tensor
            The computed loss value.
        """
        targets = targets[:, :, 0]
        targets = KerasBackend.one_hot(tf.cast(targets, tf.int32), self.config.n_classes + 1)
        targets = targets[:, :, :-1]
        outputs = self._reshape_outputs(outputs)
        return KerasBackend.categorical_crossentropy(targets, outputs)

    def _ious(self, targets, outputs):
        """
        iou = and / or
        """
        metrics = []
        targets = targets[:, :, 0]
        for c in range(self.config.n_classes):
            outs = self._eq_pixel(outputs, c)
            tars = self._eq_pixel(targets, c)
            divs = self._total_pixel(tars & outs) / self._total_pixel(tars | outs)
            metrics.append(KerasBackend.mean(divs[(self._total_pixel(tars) > 0)]))

        return tf.stack(metrics)

    def _precision_outputs(self, targets, outputs, valids):
        outputs = self._reshape_outputs(outputs)
        outputs = KerasBackend.reshape(outputs, (-1, self.config.n_classes))
        outputs = self._top(outputs)
        outs = valids & self._eq_pixel(targets, outputs)
        outs = tf.cast(outs, tf.float32)
        return KerasBackend.sum(outs)

    def _precision_targets(self, targets):
        targets = tf.cast(KerasBackend.flatten(targets), tf.int64)
        valids = ~self._eq_pixel(targets, self.config.n_classes)
        refs = tf.cast(valids, tf.float32)
        return KerasBackend.sum(refs), valids, targets

    def _reshape_outputs(self, outputs):
        return tf.reshape(outputs, [tf.shape(outputs)[0], -1, outputs.shape[-1]])

    def _total_pixel(self, tensor):
        return KerasBackend.sum(tf.cast(tensor, tf.int32), axis=1)

    def _eq_pixel(self, a, b):
        return KerasBackend.equal(a, b)

    def _top(self, t):
        return KerasBackend.argmax(t, axis=-1)

    def plot(self, cm, classes):
        """
        Plot a confusion matrix.

        This function visualizes a confusion matrix using a heatmap.
        The matrix can be normalized or displayed as raw counts.

        Parameters:
        -----------
        cm : np.ndarray
            Confusion matrix as a 2D numpy array where `cm[i, j]`
            represents the number of samples with true label `i` and
            predicted label `j`.
        classes : list of str
            List of class names corresponding to the labels in the confusion
            matrix.

        Returns:
        --------
            None
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(
            include_values=False, cmap=plt.cm.Greens, ax=ax, xticks_rotation="vertical"
        )

        ax.set_title("Confusion matrix", fontsize=11)
        ax.set_xlabel("Predicted label", fontsize=9)
        ax.set_ylabel("True label", fontsize=9)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], ".2f"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7,
                )

        plt.tight_layout()



import numpy as np
import copy
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from mct_quantizers import (
    KerasActivationQuantizationHolder,
    KerasQuantizationWrapper,
)


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # GPUs visible to tensorflow
    try:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)

class KerasSegmentationValidator:
    def __init__(self, config):
        self.ini = config

    def validate(self, dataloader_eval):
        self.dataloader_eval = dataloader_eval
        self.run()
        return None


    def calculate_iou(self, test_generator, model_path, image_size, nb_classes):
        label = np.zeros((len(test_generator), np.prod(image_size)), dtype="float32")
        X = np.zeros(
            (len(test_generator), image_size[0], image_size[1], 3),
            dtype="float32",
        )
        for n in tqdm(range(len(test_generator))):
            x, y, _ = test_generator.__getitem__(n)
            label[n, :] = y[0, :, 0]
            X[n, :, :, :] = x

        model = keras.models.load_model(
            model_path,
            custom_objects={
                "KerasActivationQuantizationHolder": KerasActivationQuantizationHolder,
                "KerasQuantizationWrapper": KerasQuantizationWrapper,
            },
            compile=False,
        )

        preds = model.predict(X, batch_size=1)
        conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
        mask = np.reshape(np.argmax(preds, axis=-1), (-1,) + image_size)
        flat_pred = np.ravel(mask).astype("int")
        flat_label = np.ravel(label).astype("int")
        for p, label in zip(flat_pred, flat_label):
            if label == nb_classes:
                continue
            if label < nb_classes and p < nb_classes:
                conf_m[label, p] += 1
            else:
                print(
                    "Invalid entry encountered, skipping! Label: ",
                    label,
                    " Prediction: ",
                    p,
                )
        intersection = np.diag(conf_m)
        U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - intersection
        IOU = intersection / U
        meanIOU = np.mean(IOU)
        return conf_m, meanIOU


    def normalize_cm(self, matrix):
        base = matrix.astype("float")
        div = matrix.sum(axis=1)
        return base / div[:, np.newaxis]

    def run(self):
        dlab = self.ini.deeplab
        config = dlab.config

        model_path = config.best_weights
        model_input_size = config.image_size
        nb_classes = config.n_classes
        conf_matrix = config.conf_matrix

        classes = [c for c in config.classes.values()][:-1]

        loader = copy.deepcopy(self.dataloader_eval)
        cm, _ = self.calculate_iou(
            loader,
            model_path,
            model_input_size,
            nb_classes,
        )
        plt.figure(figsize=(12, 8))
        plt.subplot(121)
        matrix = self.normalize_cm(cm)

        dlab.plot(matrix, classes)
        miou = np.round(np.diag(matrix).mean(), 2)
        plt.title("Trained model \nMean IOU: " + str(miou))
        plt.savefig(conf_matrix)

        print("mean IOU ", miou)
        dumps = self.ini.results.valid
        dumps["mean IOU"] = miou

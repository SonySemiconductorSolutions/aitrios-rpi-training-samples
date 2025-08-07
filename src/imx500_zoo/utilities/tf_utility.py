import os
from io import StringIO
from contextlib import redirect_stdout

import tensorflow as tf


def summary(model, f_log):
    with StringIO() as buffer, redirect_stdout(buffer):
        model.summary()
        summary_string = buffer.getvalue()

    with open(f_log, 'w') as f:
        f.write(summary_string)

def cuda_visible(n = -1):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{n}"

def physical_gpus():
    """
    https://www.tensorflow.org/guide/gpu
    """
    return tf.config.list_physical_devices("GPU")

def logical_gpus():
    return tf.config.list_logical_devices("GPU")

def assign_gpu(n=0):
    # Assign the GPU for training
    gpus = physical_gpus()
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[n], "GPU")
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)

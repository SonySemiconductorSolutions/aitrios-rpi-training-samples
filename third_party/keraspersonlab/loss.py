import tensorflow as tf
from keras import layers as KL
from keras import backend as KB
from keras import models as KM
from keras import losses

from .post_processing import Post_Process
import numpy as np

post_proc = None

config = None

def set_config(config_i):
    global config
    global post_proc
    
    config = config_i
    post_proc = Post_Process()

def tf_repeat(tensor, repeats):
    """
    From  https://github.com/tensorflow/tensorflow/issues/8246

    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.compat.v1.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tensor


def kp_map_loss_fn(x, kp_maps_pred):
    global kp_maps_true
    kp_maps_true = x[:, :, :, 0:config.NUM_KP]
    kp_maps_pred = post_proc.tensor_post_processing(kp_maps_pred, 1)
    loss = KB.mean(KB.binary_crossentropy(kp_maps_true, kp_maps_pred), axis=-1, keepdims=True)
    return KB.mean(loss, keepdims=True) * config.LOSS_WEIGHTS['heatmap']



def short_offset_loss_fn(x, short_offset_pred):

    short_offset_true = x
    short_offset_pred = post_proc.tensor_post_processing(short_offset_pred, 2)
    loss = KB.abs(short_offset_pred - short_offset_true) / config.KP_RADIUS
    loss = loss * tf_repeat(kp_maps_true, [1, 1, 1, 2])
    loss = KB.sum(loss, keepdims=True) / (KB.sum(kp_maps_true) + KB.epsilon())
    return loss * config.LOSS_WEIGHTS['short']



def mid_offset_loss_fn(x, mid_offset_pred):

    mid_offset_true = x

    mid_offset_pred = post_proc.tensor_post_processing(mid_offset_pred, 3)
    post_proc.reset()

    loss = KB.abs(mid_offset_pred - mid_offset_true) / config.KP_RADIUS
    reordered_maps = []
    for mid_idx, edge in enumerate(config.EDGES + [edge[::-1] for edge in config.EDGES]):
        from_kp = edge[0]
        reordered_maps.extend([kp_maps_true[:,:,:,from_kp], kp_maps_true[:,:,:,from_kp]])
    reordered_maps = KB.stack(reordered_maps, axis=-1)
    loss = loss * reordered_maps
    loss = KB.sum(loss, keepdims=True) / (KB.sum(reordered_maps)+KB.epsilon())

    return loss * config.LOSS_WEIGHTS['mid']

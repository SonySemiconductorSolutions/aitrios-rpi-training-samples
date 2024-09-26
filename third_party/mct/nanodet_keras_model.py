# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Nanodet-Plus Object Detection Model

This code contains a TensorFlow/Keras implementation of Nanodet-Plus object detection model, following
https://github.com/RangiLyu/nanodet. This implementation also integrates the Nanodet-plus post-processing into the
model.

The Nanodet-Plus model is optimized for real-time and resource-constrained environments while maintaining competitive
detection performance. It is particularly suitable for edge devices and embedded systems.

The code is organized as follows:
- Function definitions for building the Nanodet-Plus model.
- Model definition

For more details on the Nanodet-Plus model, refer to the original repository:
https://github.com/RangiLyu/nanodet

"""

import numpy as np
from keras.utils import plot_model
from keras.utils import get_source_inputs
from keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    DepthwiseConv2D,
    Concatenate,
    Add,
    ZeroPadding2D,
    LeakyReLU,
    Resizing,
    ReLU,
    Softmax,
)
from keras.models import Model
import keras.backend as K
import tensorflow as tf


CLASSES_DEFAULT = 80  # min(91 - 1, 80) = min(coco2017 - (not include background category), nanodet-spec)
_nanodet_classes = CLASSES_DEFAULT


def set_nanodet_classes(num=CLASSES_DEFAULT):
    global _nanodet_classes
    _nanodet_classes = num


def classes_print():
    print(f"\r\n classes : {_nanodet_classes} \r\n")


# Nanodet-Plus building blocks
def _channel_split(x):
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = x[:, :, :, 0:ip]
    c = x[:, :, :, ip:]
    return c_hat, c


def _channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = tf.reshape(x, [-1, height, width, 2, channels_per_split])
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, height, width, channels])
    return x


def _shuffle_unit(
    inputs, out_channels, bottleneck_ratio, strides=2, stage=1, block=1
):
    if K.image_data_format() == "channels_last":
        bn_axis = -1
    else:
        raise ValueError("Only channels last supported")

    prefix = "backbone.stage{}.{}".format(stage, block - 1)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = _channel_split(inputs)
        inputs = c

    x = Conv2D(
        bottleneck_channels,
        kernel_size=(1, 1),
        strides=1,
        padding="same",
        use_bias=False,
        name="{}.branch2.0".format(prefix),
    )(inputs)
    x = BatchNormalization(
        axis=bn_axis, epsilon=1e-05, name="{}.branch2.1".format(prefix)
    )(x)
    x = LeakyReLU(alpha=0.1, name="{}.branch2.2".format(prefix))(x)
    if strides > 1:
        x = ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
        x = DepthwiseConv2D(
            kernel_size=3,
            strides=strides,
            padding="valid",
            use_bias=False,
            name="{}.branch2.3".format(prefix),
        )(x)
    else:
        x = DepthwiseConv2D(
            kernel_size=3,
            strides=strides,
            padding="same",
            use_bias=False,
            name="{}.branch2.3".format(prefix),
        )(x)
    x = BatchNormalization(
        axis=bn_axis, epsilon=1e-05, name="{}.branch2.4".format(prefix)
    )(x)
    x = Conv2D(
        bottleneck_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name="{}.branch2.5".format(prefix),
    )(x)
    x = BatchNormalization(
        axis=bn_axis, epsilon=1e-05, name="{}.branch2.6".format(prefix)
    )(x)
    x = LeakyReLU(alpha=0.1, name="{}.branch2.7".format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name="{}/concat_1".format(prefix))(
            [c_hat, x]
        )
    else:
        inputs = ZeroPadding2D(padding=((1, 0), (1, 0)))(inputs)
        s2 = DepthwiseConv2D(
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            name="{}.branch1.0".format(prefix),
        )(inputs)
        s2 = BatchNormalization(
            axis=bn_axis, epsilon=1e-05, name="{}.branch1.1".format(prefix)
        )(s2)
        s2 = Conv2D(
            bottleneck_channels,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name="{}.branch1.2".format(prefix),
        )(s2)
        s2 = BatchNormalization(
            axis=bn_axis, epsilon=1e-05, name="{}.branch1.3".format(prefix)
        )(s2)
        s2 = LeakyReLU(alpha=0.1, name="{}.branch1.4".format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name="{}/concat_2".format(prefix))(
            [s2, x]
        )

    ret = _channel_shuffle(ret)
    return ret


def _block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = _shuffle_unit(
        x,
        out_channels=channel_map[stage - 1],
        strides=2,
        bottleneck_ratio=bottleneck_ratio,
        stage=stage,
        block=1,
    )

    for i in range(1, repeat + 1):
        x = _shuffle_unit(
            x,
            out_channels=channel_map[stage - 1],
            strides=1,
            bottleneck_ratio=bottleneck_ratio,
            stage=stage,
            block=(1 + i),
        )

    return x


def _nanodet_shufflenet_v2(
    input_tensor=None,
    scale_factor=1.5,
    input_shape=(416, 416, 3),
    num_shuffle_units=[3, 7, 3],
    bottleneck_ratio=0.5,
):
    if scale_factor == 1.0:
        out_channels_in_stage = np.array([24, 116, 232, 464, 1024])
    else:  # scale_factor == 1.5:
        out_channels_in_stage = np.array([24, 176, 352, 704, 1024])

    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # create shufflenet architecture
    x = ZeroPadding2D(padding=((1, 0), (1, 0)))(img_input)
    x = Conv2D(
        filters=out_channels_in_stage[0],
        kernel_size=(3, 3),
        padding="valid",
        use_bias=False,
        strides=(2, 2),
        name="backbone.conv1.0",
    )(x)
    x = BatchNormalization(epsilon=1e-05, name="backbone.conv1.1")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool1"
    )(x)

    # create stages containing shufflenet units beginning at stage 2
    out = []
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = _block(
            x,
            out_channels_in_stage,
            repeat=repeat,
            bottleneck_ratio=bottleneck_ratio,
            stage=stage + 2,
        )
        out.append(x)

    if input_tensor:
        inputs = get_source_inputs(input_tensor)

    else:
        inputs = img_input

    return inputs, out


def _conv_module(x, out_channels, kernel, name_prefix="ConvModule"):
    x = Conv2D(
        out_channels, kernel, use_bias=False, name=name_prefix + ".conv"
    )(x)
    x = BatchNormalization(epsilon=1e-05, name=name_prefix + ".bn")(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def _ghost_blocks(x, out_channels=128, name_prefix="GhostBlocks"):
    residual = x
    x1 = Conv2D(
        out_channels // 2,
        1,
        use_bias=False,
        name=name_prefix + ".ghost1.primary_conv.0",
    )(x)
    x1 = BatchNormalization(
        epsilon=1e-05, name=name_prefix + ".ghost1.primary_conv.1"
    )(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)
    x2 = DepthwiseConv2D(
        3,
        padding="same",
        use_bias=False,
        name=name_prefix + ".ghost1.cheap_operation.0",
    )(x1)
    x2 = BatchNormalization(
        epsilon=1e-05, name=name_prefix + ".ghost1.cheap_operation.1"
    )(x2)
    x2 = LeakyReLU(alpha=0.1)(x2)
    x = Concatenate()([x1, x2])

    x1 = Conv2D(
        out_channels // 2,
        1,
        use_bias=False,
        name=name_prefix + ".ghost2.primary_conv.0",
    )(x)
    x1 = BatchNormalization(
        epsilon=1e-05, name=name_prefix + ".ghost2.primary_conv.1"
    )(x1)
    x2 = DepthwiseConv2D(
        3,
        padding="same",
        use_bias=False,
        name=name_prefix + ".ghost2.cheap_operation.0",
    )(x1)
    x2 = BatchNormalization(
        epsilon=1e-05, name=name_prefix + ".ghost2.cheap_operation.1"
    )(x2)
    x = Concatenate()([x1, x2])

    residual = DepthwiseConv2D(
        5, padding="same", use_bias=False, name=name_prefix + ".shortcut.0"
    )(residual)
    residual = BatchNormalization(
        epsilon=1e-05, name=name_prefix + ".shortcut.1"
    )(residual)
    residual = Conv2D(
        out_channels, 1, use_bias=False, name=name_prefix + ".shortcut.2"
    )(residual)
    residual = BatchNormalization(
        epsilon=1e-05, name=name_prefix + ".shortcut.3"
    )(residual)

    x = Add()([residual, x])
    return x


def _depthwise_conv_module(
    x, out_channels=128, stride=2, name_prefix="DepthwiseConvModule"
):
    if stride > 1:
        x = ZeroPadding2D(padding=((2, 2), (2, 2)))(x)
        x = DepthwiseConv2D(
            5,
            strides=(stride, stride),
            padding="valid",
            use_bias=False,
            name=name_prefix + ".depthwise",
        )(x)
    else:
        x = DepthwiseConv2D(
            5,
            strides=(stride, stride),
            padding="same",
            use_bias=False,
            name=name_prefix + ".depthwise",
        )(x)
    x = BatchNormalization(epsilon=1e-05, name=name_prefix + ".dwnorm")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(
        out_channels, 1, use_bias=False, name=name_prefix + ".pointwise"
    )(x)
    x = BatchNormalization(epsilon=1e-05, name=name_prefix + ".pwnorm")(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def _nanodet_ghostpan(
    x, in_channels=[176, 352, 704], out_channels=128, res=416
):

    for idx in range(len(in_channels)):
        x[idx] = _conv_module(
            x[idx],
            out_channels,
            1,
            name_prefix="fpn.reduce_layers." + str(idx),
        )

    # top-down path
    p4 = x[2]
    x_upsampled = Resizing(
        int(res / 16), int(res / 16), interpolation="bilinear"
    )(x[2])
    x_concate = Concatenate(axis=-1, name="p3_input")([x_upsampled, x[1]])
    p3 = _ghost_blocks(
        x_concate, out_channels, name_prefix="fpn.top_down_blocks.0.blocks.0"
    )

    x_upsampled = Resizing(
        int(res / 8), int(res / 8), interpolation="bilinear"
    )(p3)
    x_concate = Concatenate(axis=-1, name="p2_input")([x_upsampled, x[0]])
    p2 = _ghost_blocks(
        x_concate, out_channels, name_prefix="fpn.top_down_blocks.1.blocks.0"
    )

    # bottom up path
    n2 = p2

    x_downsampled = _depthwise_conv_module(
        n2, out_channels, name_prefix="fpn.downsamples.0"
    )
    x_concate = Concatenate(axis=-1, name="n3_input")([x_downsampled, p3])
    n3 = _ghost_blocks(
        x_concate, out_channels, name_prefix="fpn.bottom_up_blocks.0.blocks.0"
    )

    x_downsampled = _depthwise_conv_module(
        n3, out_channels, name_prefix="fpn.downsamples.1"
    )
    x_concate = Concatenate(axis=-1, name="n4_input")([x_downsampled, p4])
    n4 = _ghost_blocks(
        x_concate, out_channels, name_prefix="fpn.bottom_up_blocks.1.blocks.0"
    )

    n5_a = _depthwise_conv_module(
        n4, out_channels, name_prefix="fpn.extra_lvl_out_conv.0"
    )
    n5_b = _depthwise_conv_module(
        p4, out_channels, name_prefix="fpn.extra_lvl_in_conv.0"
    )

    n5 = Add()([n5_a, n5_b])
    return [n2, n3, n4, n5]


def _distance2bbox(points, distance):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    d0, d1, d2, d3 = tf.unstack(distance, 4, -1)
    a0, a1, a2, a3 = tf.unstack(points, 4, -1)
    x1 = tf.math.add(a0, -d0)
    y1 = tf.math.add(a1, -d1)
    x2 = tf.math.add(a0, d2)
    y2 = tf.math.add(a1, d3)
    return x1, y1, x2, y2


def _dfl(x, c1=8):
    """Distributed focal loss calculation.
    Args:
        c1:
        x:

    Returns:
        Tensor: bboxes after integral calculation.
    """
    x_shape = x.shape
    x = tf.reshape(x, [-1, x_shape[1] * x_shape[2], 4 * c1])
    x = tf.reshape(x, [-1, x_shape[1] * x_shape[2], 4, c1])
    x = Softmax(-1)(x)
    w = np.expand_dims(np.expand_dims(np.expand_dims(np.arange(c1), 0), 0), -1)
    conv = Conv2D(1, 1, use_bias=False, weights=[w])
    return tf.squeeze(conv(x), -1)


def _nanodet_generate_anchors(batch_size, featmap_sizes, strides):
    anchors_list = []
    for i, stride in enumerate(strides):
        h, w = featmap_sizes[i]
        x_range = np.arange(w) * stride
        y_range = np.arange(h) * stride
        y, x = np.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = np.ones_like(x) * stride
        anchors = np.stack([y, x, strides, strides], axis=-1)
        anchors = np.expand_dims(anchors, axis=0)
        anchors = np.repeat(anchors, batch_size, axis=0)
        anchors_list.append(anchors)
    return np.concatenate(anchors_list, axis=1, dtype=float)


def nanodet_plus_head(n, feat_channels=128):
    h = n
    for idx in range(4):
        h[idx] = _depthwise_conv_module(
            n[idx],
            out_channels=feat_channels,
            stride=1,
            name_prefix="head.cls_convs." + str(idx) + ".0",
        )
        h[idx] = _depthwise_conv_module(
            h[idx],
            out_channels=feat_channels,
            stride=1,
            name_prefix="head.cls_convs." + str(idx) + ".1",
        )
        h[idx] = Conv2D(
            _nanodet_classes + 32, 1, name="head.gfl_cls." + str(idx)
        )(h[idx])
    return h


def nanodet_box_decoding(h, res):
    strides = [8, 16, 32, 64]
    batch_size = 1
    featmap_sizes = [
        (np.ceil(res / stride), np.ceil(res / stride)) for stride in strides
    ]
    all_anchors = (
        1 / res * _nanodet_generate_anchors(batch_size, featmap_sizes, strides)
    )
    nn = res / 8 * res / 8
    anchors_list = np.split(
        all_anchors, [int(nn), int(1.25 * nn), int(1.3125 * nn)], axis=1
    )
    h_cls = []
    h_bbox = []
    for idx in range(4):
        # Split to 80 classes and 4 * 8 bounding boxes regression
        cls, regr = tf.split(h[idx], [_nanodet_classes, 32], -1)
        ndet = cls.shape[1] * cls.shape[2]

        # Distributed Focal loss integral
        d = _dfl(regr, 8)

        # Box decoding
        anchors = tf.constant(anchors_list[idx], dtype=tf.float32)
        d = tf.math.multiply(d, anchors[..., 2, None])
        bbox0, bbox1, bbox2, bbox3 = _distance2bbox(anchors, d)
        bbox0, bbox1, bbox2, bbox3 = (
            ReLU()(bbox0),
            ReLU()(bbox1),
            ReLU()(bbox2),
            ReLU()(bbox3),
        )
        bbox = tf.stack([bbox1, bbox0, bbox3, bbox2], -1)
        bbox = tf.expand_dims(bbox, 2)

        cls = tf.reshape(cls, [-1, ndet, _nanodet_classes])
        h_cls.append(cls)
        h_bbox.append(bbox)
    classes = Concatenate(axis=1)([h_cls[0], h_cls[1], h_cls[2], h_cls[3]])
    boxes = Concatenate(axis=1)([h_bbox[0], h_bbox[1], h_bbox[2], h_bbox[3]])
    classes = tf.math.sigmoid(classes)
    return classes, boxes


# Nanodet-Plus model definition
def nanodet_plus_m(input_shape, scale_factor, bottleneck_ratio, feat_channels):
    """
    Create the Nanodet-Plus object detection model.

    Args:
        input_shape (tuple): The shape of input images (height, width, channels).
        scale_factor (float): Scale factor for ShuffleNetV2 backbone.
        bottleneck_ratio (float): Bottleneck ratio for ShuffleNetV2 backbone.
        feat_channels (int): Number of feature channels.

    Returns:
        tf.keras.Model: The Nanodet-Plus model.

    Configuration options:
        nanodet-plus-m-1.5x-416:  input_shape = (416,416,3), scale_factor=1.5, feat_channels=128
        nanodet-plus-m-1.5x-320:  input_shape = (320,320,3), scale_factor=1.5, feat_channels=128
        nanodet-plus-m-416:  input_shape = (416,416,3), scale_factor=1.0, feat_channels=96
        nanodet-plus-m-320:  input_shape = (320,320,3), scale_factor=1.0, feat_channels=96

    """
    # Nanodet backbone
    inputs, x = _nanodet_shufflenet_v2(
        scale_factor=scale_factor,
        input_shape=input_shape,
        bottleneck_ratio=bottleneck_ratio,
    )

    # Nanodet neck
    x = _nanodet_ghostpan(x, out_channels=feat_channels, res=input_shape[0])

    # Nanodet head
    x = nanodet_plus_head(x, feat_channels=feat_channels)

    # Define Keras model
    return Model(
        inputs, x, name=f"Nanodet_plus_m_{scale_factor}x_{input_shape[0]}"
    )

import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D, Conv2DTranspose
from keras.layers import GlobalAveragePooling2D

DILATION = True
CONV2DTRANSPOSE = True


class DeepLabArchitecture:
    def __init__(self, **kwargs) -> None:
        """If you want to initialize and parameterize the
        custom class using custom kwargs
        """
        pass

    def _make_divisible(self, v, divisor, min_value=None):
        """
        Adjusts a value to be divisible by a given divisor,
        ensuring that it meets a minimum value constraint.

        This function rounds the input value `v` to the nearest
        multiple of the `divisor`.
        If the adjusted value is less than 90% of the original value,
        it is incremented by the divisor.
        Additionally, the adjusted value is constrained by a
        minimum value if provided.

        Parameters:
        -----------
        v : float or int
            The value to be adjusted to be divisible by the divisor.
        divisor : int
            The value to which `v` should be made divisible.
        min_value : int, optional
            The minimum value that the adjusted result should meet.
            If None, the divisor is used as the minimum value.

        Returns:
        --------
        int
            The adjusted value, which is divisible by the divisor and
            meets the minimum value constraint.
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Ensure that rounding down does not result in a value less than
        # 90% of the original value
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _inverted_res_block(
        self,
        inputs,
        expansion,
        stride,
        alpha,
        filters,
        block_id,
        skip_connection,
        rate=1,
    ):
        """
        Builds an inverted residual block with optional skip connection.

        This function implements a block commonly used in MobileNetV2
        architectures. It consists of an optional expansion layer
        (1x1 convolution), a depthwise separable convolution with an
        optional dilation rate, and a projection layer (another 1x1
        convolution). The block can optionally use a skip connection,
        adding the input to the output, which is particularly useful in
        deeper networks to prevent the vanishing gradient problem.

        Parameters:
        -----------
        inputs : tensor
            Input tensor of shape (batch_size, height, width, channels).
        expansion : int
            Factor by which the input channels are expanded in the expansion
            layer.
        stride : int
            Stride for the depthwise convolution. Typically 1 or 2.
        alpha : float
            Width multiplier to reduce the number of filters in the depthwise
            and pointwise convolutions.
        filters : int
            Number of filters in the pointwise convolution (before multiplying
            by alpha).
        block_id : int
            Unique identifier for the block. Used to create layer names.
        skip_connection : bool
            If True, adds the input tensor to the output tensor.
        rate : int, optional, default=1
            Dilation rate for the depthwise convolution. If greater than 1,
            a transposed convolution
            is applied before the depthwise convolution.

        Returns:
        --------
        tensor
            Output tensor after applying the block's operations.
        """
        in_channels = inputs.shape[-1]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = pointwise_conv_filters
        x = inputs
        prefix = "expanded_conv_{}_".format(block_id)
        if block_id:
            # Expand
            x = Conv2D(
                expansion * in_channels,
                kernel_size=1,
                padding="same",
                use_bias=False,
                activation=None,
                name=prefix + "expand",
            )(x)
            x = BatchNormalization(
                epsilon=1e-3,
                momentum=0.999,
                name=prefix + "expand_BN",
            )(x)
            x = Activation("relu", name=prefix + "expand_relu")(x)
        else:
            prefix = "expanded_conv_"
        # Depthwise
        # ADDED
        if rate > 1 and CONV2DTRANSPOSE:
            x = Conv2DTranspose(
                filters,
                3,
                strides=(1, 1),
                dilation_rate=(1, 1),
                padding="valid",
                use_bias=False,
                name=prefix + "_transpose",
            )(x)
        else:
            x = DepthwiseConv2D(
                kernel_size=3,
                strides=stride,
                activation=None,
                use_bias=False,
                padding="same",
                dilation_rate=(rate, rate),
                name=prefix + "depthwise",
            )(x)
        x = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name=prefix + "depthwise_BN"
        )(x)

        x = Activation("relu", name=prefix + "depthwise_relu")(x)

        # Project
        x = Conv2D(
            pointwise_filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            activation=None,
            name=prefix + "project",
        )(x)

        # ADDED (make sure the shapes are correct)
        if rate > 1 and CONV2DTRANSPOSE:
            x = DepthwiseConv2D(
                kernel_size=3,
                strides=stride,
                activation=None,
                use_bias=False,
                padding="valid",
                name=prefix + "project_extra_depthwise",
            )(x)

        x = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name=prefix + "project_BN"
        )(x)

        if skip_connection:
            return Add(name=prefix + "add")([inputs, x])
        return x

    def Deeplabv3(
        self,
        weights=None,
        input_tensor=None,
        input_shape=None,
        classes=21,
        backbone="mobilenetv2",
        OS=16,
        alpha=1.0,
        activation=None,
    ):
        """Instantiates the Deeplabv3+ architecture
        Optionally loads weights pre-trained
        on PASCAL VOC or Cityscapes. This model is available for TensorFlow
        only.
        # Arguments
            weights: one of 'pascal_voc' (pre-trained on pascal voc),
                'cityscapes' (pre-trained on cityscape) or None (random
                initialization)
            input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`) to use as image input for the model.
            input_shape: shape of input image. format HxWxC
                PASCAL VOC model was trained on (512,512,3) images.
                None is allowed as shape/width
            classes: number of desired classes. PASCAL VOC has 21 classes,
            Cityscapes has 19 classes. If number of classes not aligned
            with the weights used, last layer is initialized randomly
            backbone: backbone to use 'mobilenetv2'.
            activation: optional activation to add to the top of the network.
                One of 'softmax', 'sigmoid' or None
            OS: determines input_shape/feature_extractor_output ratio.
            One of {8,16}. Used only for xception backbone.
            alpha: controls the width of the MobileNetV2 network.
            This is known as the width multiplier in the MobileNetV2 paper.
                    - If `alpha` < 1.0, proportionally decreases the number
                        of filters in each layer.
                    - If `alpha` > 1.0, proportionally increases the number
                        of filters in each layer.
                    - If `alpha` = 1, default number of filters from the paper
                        are used at each layer.
                Used only for mobilenetv2 backbone.
                Pretrained is only available for alpha=1.
        # Returns
            A Keras model instance.
        # Raises
            RuntimeError: If attempting to run this model with a
                backend that does not support separable convolutions.
            ValueError: in case of invalid argument for `weights` or
            `backbone`
        """
        if input_tensor is None:
            img_input = Input(shape=input_shape)

        first_block_filters = self._make_divisible(16 * alpha, 8)

        x = Conv2D(
            first_block_filters,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            name="Conv" if input_shape[2] == 3 else "Conv_",
        )(img_input)

        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="Conv_BN")(x)

        x = Activation("relu", name="Conv_Relu6")(x)

        x = self._inverted_res_block(
            x,
            filters=8,
            alpha=alpha,
            stride=1,
            expansion=1,
            block_id=0,
            skip_connection=False,
        )

        x = self._inverted_res_block(
            x,
            filters=12,
            alpha=alpha,
            stride=1,
            expansion=6,
            block_id=1,
            skip_connection=False,
        )

        x = self._inverted_res_block(
            x,
            filters=12,
            alpha=alpha,
            stride=1,
            expansion=6,
            block_id=2,
            skip_connection=True,
        )

        x = self._inverted_res_block(
            x,
            filters=16,
            alpha=alpha,
            stride=1,
            expansion=6,
            block_id=3,
            skip_connection=False,
        )
        x = self._inverted_res_block(
            x,
            filters=16,
            alpha=alpha,
            stride=1,
            expansion=6,
            block_id=4,
            skip_connection=True,
        )
        x = self._inverted_res_block(
            x,
            filters=16,
            alpha=alpha,
            stride=1,
            expansion=6,
            block_id=5,
            skip_connection=True,
        )

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = self._inverted_res_block(
            x,
            filters=32,
            alpha=alpha,
            stride=1,  # 1!
            expansion=6,
            block_id=6,
            skip_connection=False,
        )
        x = self._inverted_res_block(
            x,
            filters=32,
            alpha=alpha,
            stride=1,
            rate=2 if DILATION else 1,
            expansion=6,
            block_id=7,
            skip_connection=True,
        )
        x = self._inverted_res_block(
            x,
            filters=32,
            alpha=alpha,
            stride=1,
            rate=2 if DILATION else 1,
            expansion=6,
            block_id=8,
            skip_connection=True,
        )
        x = self._inverted_res_block(
            x,
            filters=32,
            alpha=alpha,
            stride=1,
            rate=2 if DILATION else 1,
            expansion=6,
            block_id=9,
            skip_connection=True,
        )

        x = self._inverted_res_block(
            x,
            filters=48,
            alpha=alpha,
            stride=1,
            rate=2 if DILATION else 1,
            expansion=6,
            block_id=10,
            skip_connection=False,
        )
        x = self._inverted_res_block(
            x,
            filters=48,
            alpha=alpha,
            stride=1,
            rate=2 if DILATION else 1,
            expansion=6,
            block_id=11,
            skip_connection=True,
        )
        x = self._inverted_res_block(
            x,
            filters=48,
            alpha=alpha,
            stride=1,
            rate=2 if DILATION else 1,
            expansion=6,
            block_id=12,
            skip_connection=True,
        )

        x = self._inverted_res_block(
            x,
            filters=80,
            alpha=alpha,
            stride=1,
            rate=2 if DILATION else 1,  # 1!
            expansion=6,
            block_id=13,
            skip_connection=False,
        )
        x = self._inverted_res_block(
            x,
            filters=80,
            alpha=alpha,
            stride=1,
            rate=4 if DILATION else 1,
            expansion=6,
            block_id=14,
            skip_connection=True,
        )
        x = self._inverted_res_block(
            x,
            filters=80,
            alpha=alpha,
            stride=1,
            rate=4 if DILATION else 1,
            expansion=6,
            block_id=15,
            skip_connection=True,
        )

        x = self._inverted_res_block(
            x,
            filters=160,
            alpha=alpha,
            stride=1,
            rate=4 if DILATION else 1,
            expansion=6,
            block_id=16,
            skip_connection=False,
        )

        # end of feature extractor
        # branching for Atrous Spatial Pyramid Pooling

        # Image Feature branch
        b4 = GlobalAveragePooling2D()(x)
        b4_shape = tf.keras.backend.int_shape(b4)

        # from (b_size, channels)->(b_size, 1, 1, channels)
        b4 = Reshape((1, 1, b4_shape[1]))(b4)

        b4 = Conv2D(
            256,
            (1, 1),
            padding="same",
            use_bias=False,
            name="image_pooling",
        )(b4)
        b4 = BatchNormalization(name="image_pooling_BN", epsilon=1e-5)(b4)
        b4 = Activation("relu")(b4)

        # upsample. have to use compat because of the option align_corners
        size_before = tf.keras.backend.int_shape(x)
        b4 = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before[1:3], interpolation="bilinear"
        )(b4)

        # simple 1x1
        b0 = Conv2D(256, (1, 1), padding="same", use_bias=False, name="aspp0")(x)
        b0 = BatchNormalization(name="aspp0_BN", epsilon=1e-5)(b0)
        b0 = Activation("relu", name="aspp0_activation")(b0)

        x = Concatenate()([b4, b0])

        x = Conv2D(
            256,
            (1, 1),
            padding="same",
            use_bias=False,
            name="concat_projection",
        )(x)
        x = BatchNormalization(name="concat_projection_BN", epsilon=1e-5)(x)
        x = Activation("relu")(x)
        x = Dropout(0.1)(x)

        # DeepLab v.3+ decoder
        # you can use it with arbitary number of classes
        if (weights == "pascal_voc" and classes == 21) or (
            weights == "cityscapes" and classes == 19
        ):
            last_layer_name = "logits_semantic"
        else:
            last_layer_name = "custom_logits_semantic"

        x = Conv2D(classes, (1, 1), padding="same", name=last_layer_name)(x)

        size_before3 = tf.keras.backend.int_shape(img_input)

        x = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before3[1:3], interpolation="bilinear"
        )(x)

        x = Activation("softmax", name="pred_mask")(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = tf.keras.utils.get_source_inputs.get_source_inputs(input_tensor)
        else:
            inputs = img_input

        model = Model(inputs, x, name="deeplabv3plus")
        return model

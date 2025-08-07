from keras import layers as KL
from keras import models as KM

from third_party.tensorflow.mobilenet import get_mobilenetv1_base


class ConfigTemplate:
    def __init__(self):
        self.OUTPUT_STRIDE = 16


config = ConfigTemplate()


def set_config(config_i):
    global config
    config = config_i


def add_personlab_head(features, img_shape, id):
    sfx = "_" + str(id)

    kp_maps = KL.Conv2D(
        config.NUM_KP, kernel_size=(1, 1), activation="sigmoid", name="kp_maps" + sfx
    )(features)
    short_offsets = KL.Conv2D(
        2 * config.NUM_KP, kernel_size=(1, 1), name="short_offsets" + sfx
    )(features)
    mid_offsets_1 = KL.Conv2D(2 * (config.NUM_EDGES), kernel_size=(1, 1))(features)
    mid_offsets_2 = KL.Conv2D(2 * (config.NUM_EDGES), kernel_size=(1, 1))(features)
    mid_offsets = KL.Concatenate(axis=-1, name="mid_offsets" + sfx)(
        [mid_offsets_1, mid_offsets_2]
    )

    outputs = [kp_maps, short_offsets, mid_offsets]
    return outputs


def get_personlab_model(
    build_base_func=get_mobilenetv1_base, output_stride=config.OUTPUT_STRIDE
):
    """
    Constructs the PersonLab model and returns the model object without compiling

    # Arguments:

        build_base_func: The function that builds the base network. The available options are get_resnet50_base and
            get_resnet101_base

    """

    img_shape = [config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]]
    input_img = KL.Input(shape=config.IMAGE_SHAPE)
    base_model = build_base_func(
        input_tensor=input_img, output_stride=output_stride, return_model=True
    )
    x = base_model(input_img)
    outputs = add_personlab_head(x, img_shape, id="head")
    model = KM.Model(inputs=input_img, outputs=outputs)
    return model



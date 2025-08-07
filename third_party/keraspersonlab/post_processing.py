import tensorflow as tf
from keras import layers as KL
from keras import backend as KB
from .bilinear import bilinear_sampler

config = None


def set_config(config_i):
    global config
    config = config_i


def refine(inputs, num_steps=2):
    base, offsets = inputs

    # sample bilinearly
    for _ in range(num_steps):
        base = base + bilinear_sampler(offsets, base)

    return base


class Post_Process:

    def __init__(self):

        self.img_shape = [config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]]
        self.reset()
        self.kp_maps_prev = 0.0
        self.short_offsets = 0.0
        self.mid_offsets = 0.0
        self.extra_input = 0.0

    @staticmethod
    def split_and_refine_mid_offsets(mid_offsets, short_offsets):

        output_mid_offsets = []
        for mid_idx, edge in enumerate(
            config.EDGES + [edge[::-1] for edge in config.EDGES]
        ):
            to_keypoint = edge[1]
            kp_short_offsets = KL.Lambda(
                lambda t: t[:, :, :, 2 * to_keypoint : 2 * to_keypoint + 2]
            )(short_offsets)
            kp_mid_offsets = KL.Lambda(
                lambda t: t[:, :, :, 2 * mid_idx : 2 * mid_idx + 2]
            )(mid_offsets)
            kp_mid_offsets = KL.Lambda(lambda t: refine(t))(
                [kp_mid_offsets, kp_short_offsets]
            )
            output_mid_offsets.append(kp_mid_offsets)

        return KL.Lambda(
            lambda t: KB.concatenate(t, axis=-1), name="mid_offsets_head"
        )(output_mid_offsets)

    def reset(self):
        self.kp_maps = 0.0
        self.short_offsets = 0.0
        self.mid_offsets = 0.0

    def previous_state(self):
        self.kp_maps_prev = self.kp_maps
        self.short_offsets = self.short_offsets
        self.mid_offsets = self.mid_offsets

    def extra_state(self, x):
        self.extra_input = x

    def tensor_post_processing(self, out_tensor, tensor_flag, sfx="PP"):

        if tensor_flag == 1:
            self.kp_maps = KL.Lambda(
                lambda t: tf.image.resize(t, self.img_shape),
                name="kp_maps" + sfx,
            )(out_tensor)
            return self.kp_maps
        elif tensor_flag == 2:
            self.short_offsets = KL.Lambda(
                lambda t: tf.image.resize(t, self.img_shape),
                name="short_offsets" + sfx,
            )(out_tensor)
            return self.short_offsets
        elif tensor_flag == 3:
            self.mid_offsets = KL.Lambda(
                lambda t: tf.image.resize(t, self.img_shape)
            )(out_tensor)
            self.mid_offsets = self.split_and_refine_mid_offsets(
                self.mid_offsets, self.short_offsets
            )
            return self.mid_offsets

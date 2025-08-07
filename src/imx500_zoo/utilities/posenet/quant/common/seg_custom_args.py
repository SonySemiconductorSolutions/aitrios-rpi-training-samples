from keras import backend as K
from keras.layers import Layer


class CustomArgmaxLayer(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(CustomArgmaxLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, mask=None):
        return K.argmax(inputs, axis=self.axis)

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super(CustomArgmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

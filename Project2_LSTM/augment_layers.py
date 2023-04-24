import tensorflow as tf
import tensorflow_io as tfio


class FreqMaskLayer(tf.keras.layers.Layer):
    def __init__(self, param, **kwargs):
        super(FreqMaskLayer, self).__init__(**kwargs)
        self.param = param

    def call(self, inputs, training=None):
        if training:
            return tf.map_fn(lambda x: tfio.audio.freq_mask(x, param=self.param), inputs)
        return inputs


class TimeMaskLayer(tf.keras.layers.Layer):
    def __init__(self, param, **kwargs):
        super(TimeMaskLayer, self).__init__(**kwargs)
        self.param = param

    def call(self, inputs, training=None):
        if training:
            return tf.map_fn(lambda x: tfio.audio.time_mask(x, param=self.param), inputs)
        return inputs

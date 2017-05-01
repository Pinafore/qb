from collections import ChainMap

from keras.layers import Layer
from keras import backend as K
import tensorflow as tf


class AverageWords(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, x, mask=None):
        axis = K.ndim(x) - 2
        if mask is not None:
            summed = K.sum(x, axis=axis)
            n_words = K.expand_dims(K.sum(K.cast(mask, 'float32'), axis=axis), axis)
            return summed / n_words
        else:
            return K.mean(x, axis=axis)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        dimensions = list(input_shape)
        n_dimensions = len(input_shape)
        del dimensions[n_dimensions - 2]
        return tuple(dimensions)


class WordDropout(Layer):
    """
    Applies Word Level Dropout to the input which helps prevent overfitting
        rate: float between 0 and 1. Fraction of the input words to drop.
    """
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.supports_masking = True

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.0:
            def dropped_inputs():
                input_shape = K.shape(inputs)
                batch_size = input_shape[0]
                n_time_steps = input_shape[1]
                mask = tf.random_uniform((batch_size, n_time_steps, 1)) >= self.rate
                w_drop = K.cast(mask, 'float32') * inputs
                return w_drop
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BatchMatmul(Layer):
    """
    Compute dot product between multiple vectors and one vector
    """
    def __init__(self, num_outputs, **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs

    def call(self, inputs):
        return tf.reduce_sum(inputs[0] * K.expand_dims(inputs[1], 1), axis=-1)

    def compute_output_shape(self, input_shape):
        return (None, self.num_outputs)

    def get_config(self):
        base_config = super().get_config()
        return dict(ChainMap(base_config, {'num_outputs': self.num_outputs}))

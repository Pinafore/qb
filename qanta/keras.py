from keras.layers import GlobalAveragePooling1D, Layer
from keras import backend as K
import tensorflow as tf


class AverageWords(GlobalAveragePooling1D):
    def call(self, x, mask=None):
        if mask is not None:
            summed = K.sum(x, axis=1)
            n_words = K.expand_dims(K.sum(K.cast(mask, 'float32'), axis=1), 1)
            average = summed / n_words
            return average
        else:
            return super().call(x)

    def compute_mask(self, inputs, mask=None):
        return None


class WordDropout(Layer):
    """Applies Word Level Dropout to the input.
    Dropout consists in randomly setting
    a fraction `rate` of input words to 0 at each update during training time,
    which helps prevent overfitting.
    # Arguments
        rate: float between 0 and 1. Fraction of the input words to drop.
        seed: A Python integer to use as random seed.
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

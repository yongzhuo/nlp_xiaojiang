# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/5/10 10:49
# @author   :Mo
# @function : 1. create model of keras-bert for get [-2] layers
#             2. create model of AttentionWeightedAverage for get avg attention pooling

from keras.engine import InputSpec
import keras.backend as k_keras
from keras.engine import Layer
from keras import initializers


class NonMaskingLayer(Layer):
    """
    fix convolutional 1D can't receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class AttentionWeightedAverage(Layer):
    '''
    codes from:  https://github.com/BrikerMan/Kashgari
    detail: https://github.com/BrikerMan/Kashgari/blob/master/kashgari/tasks/classification/models.py
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    '''

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_w'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = k_keras.dot(x, self.W)
        x_shape = k_keras.shape(x)
        logits = k_keras.reshape(logits, (x_shape[0], x_shape[1]))
        ai = k_keras.exp(logits - k_keras.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = k_keras.cast(mask, k_keras.floatx())
            ai = ai * mask
        att_weights = ai / (k_keras.sum(ai, axis=1, keepdims=True) + k_keras.epsilon())
        weighted_input = x * k_keras.expand_dims(att_weights)
        result = k_keras.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

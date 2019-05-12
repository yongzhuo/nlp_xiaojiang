# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/5/8 20:04
# @author   :Mo
# @function :embedding of bert keras

from conf.feature_config import gpu_memory_fraction, config_name, ckpt_name, vocab_file, max_seq_len, layer_indexes
from FeatureProject.bert.layers_keras import NonMaskingLayer
from keras_bert import load_trained_model_from_checkpoint
import keras.backend.tensorflow_backend as ktf_keras
import keras.backend as k_keras
from keras.models import Model
import tensorflow as tf
import os

import logging as logger
# 全局使用，使其可以django、flask、tornado等调用
graph = None
model = None

# gpu配置与使用率设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
sess = tf.Session(config=config)
ktf_keras.set_session(sess)

class KerasBertEmbedding():
    def __init__(self):
        self.config_path, self.checkpoint_path, self.dict_path, self.max_seq_len = config_name, ckpt_name, vocab_file, max_seq_len

    def bert_encode(self):
        # 全局使用，使其可以django、flask、tornado等调用
        global graph
        graph = tf.get_default_graph()
        global model
        model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path,
                                                   seq_len=self.max_seq_len)
        print(model.output)
        # 分类如果只选一层，就只取最后那一层的weight
        if len(layer_indexes) == 1:
            encoder_layer = model.get_layer(index=len(model.layers)-1).output
        # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
        else:
            # layer_indexes must be [1,2,3,......12]
            all_layers = [model.get_layer(index=lay).output for lay in layer_indexes]
            encoder_layer = k_keras.concatenate(all_layers, -1)
        print("KerasBertEmbedding:")
        print(encoder_layer.shape)
        output_layer = NonMaskingLayer()(encoder_layer)
        model = Model(model.inputs, output_layer)
        # model.summary(120)
        return model.inputs, model.output


if __name__ == "__main__":
    bert_vector = KerasBertEmbedding()
    pooled = bert_vector.bert_encode()

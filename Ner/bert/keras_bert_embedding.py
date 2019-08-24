# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/5/8 20:04
# @author   :Mo
# @function :embedding of bert keras

import os

import keras.backend.tensorflow_backend as ktf_keras
import tensorflow as tf
from Ner.bert.keras_bert_layer import NonMaskingLayer
from keras.layers import Add, Concatenate
from keras.models import Model
from keras_bert import load_trained_model_from_checkpoint

from Ner.bert.args import gpu_memory_fraction, max_seq_len, layer_indexes
from conf.feature_config import config_name, ckpt_name, vocab_file

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
        bert_layers = model.layers
        # return model
        print(bert_layers)
        print(model.output)
        print(len(model.layers))
        # lay = model.layers
        #一共104个layer，其中前八层包括token,pos,embed等，
        # 每8层（MultiHeadAttention,Dropout,Add,LayerNormalization）
        # 一共12层+最开始未处理那层(可以理解为input)
        layer_dict = [7]
        layer_0 = 7
        for i in range(12):
            layer_0 = layer_0 + 8
            layer_dict.append(layer_0)

        # 输出它本身
        if len(layer_indexes) == 0:
            encoder_layer = model.output
        # 分类如果只有一层，就只取最后那一层的weight，取得不正确
        elif len(layer_indexes) == 1:
            if layer_indexes[0] in [i+1 for i in range(13)]:
                encoder_layer = model.get_layer(index=layer_dict[layer_indexes[0]]).output
            else:
                encoder_layer = model.get_layer(index=layer_dict[-1]).output
        # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
        else:
            # layer_indexes must be [1,2,3,......12]
            # all_layers = [model.get_layer(index=lay).output if lay is not 1 else model.get_layer(index=lay).output[0] for lay in layer_indexes]
            all_layers = [model.get_layer(index=layer_dict[lay]).output if lay in [i for i in range(13)]
                          else model.get_layer(index=layer_dict[-1]).output  #如果给出不正确，就默认输出最后一层
                          for lay in layer_indexes]
            print(layer_indexes)
            print(all_layers)
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            # encoder_layer = Add()(all_layers_select)
            encoder_layer = Concatenate(axis=-1)(all_layers_select)
            print(encoder_layer.shape)
        print("KerasBertEmbedding:")
        print(encoder_layer.shape)
        output = NonMaskingLayer()(encoder_layer)
        model = Model(model.inputs, output)
        # model.summary(120)
        return model.inputs, model.output


if __name__ == "__main__":
    bert_vector = KerasBertEmbedding()
    pooled = bert_vector.bert_encode()

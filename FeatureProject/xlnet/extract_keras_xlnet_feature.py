# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/8/27 22:27
# @author   :Mo
# @function :



from keras_xlnet import Tokenizer, ATTENTION_TYPE_BI, ATTENTION_TYPE_UNI
from keras_xlnet import load_trained_model_from_checkpoint

from FeatureProject.bert.layers_keras import NonMaskingLayer
import keras.backend.tensorflow_backend as ktf_keras
from keras.models import Model
from keras.layers import Add
import tensorflow as tf
import numpy as np
import codecs
import os

from FeatureProject.xlnet import args


# 全局使用，使其可以django、flask、tornado等调用
graph = None
model = None
# gpu配置与使用率设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
sess = tf.Session(config=config)
ktf_keras.set_session(sess)


class KerasXlnetVector():
    def __init__(self):
        self.attention_type = ATTENTION_TYPE_BI if args.attention_type[0] == 'bi' else ATTENTION_TYPE_UNI
        self.memory_len, self.target_len, self.batch_size = args.memory_len, args.target_len, args.batch_size
        self.checkpoint_path, self.config_path = args.ckpt_name, args.config_name
        self.layer_indexes, self.in_train_phase = args.layer_indexes, False

        print("load KerasXlnetEmbedding start! ")
        # 全局使用，使其可以django、flask、tornado等调用
        global graph
        graph = tf.get_default_graph()
        global model
        # 模型加载
        model = load_trained_model_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                   attention_type=self.attention_type,
                                                   in_train_phase=self.in_train_phase,
                                                   config_path=self.config_path,
                                                   memory_len=self.memory_len,
                                                   target_len=self.target_len,
                                                   batch_size=self.batch_size,
                                                   mask_index=0)
        # 字典加载
        self.tokenizer = Tokenizer(args.spiece_model)
        # debug时候查看layers
        self.model_layers = model.layers
        len_layers = self.model_layers.__len__()
        print(len_layers)
        len_couche = int((len_layers - 6) / 10)
        # 一共246个layer
        # 每层10个layer（MultiHeadAttention,Dropout,Add,LayerNormalization）,第一是9个layer的输入和embedding层
        # 一共24层
        layer_dict = [5]
        layer_0 = 6
        for i in range(len_couche):
            layer_0 = layer_0 + 10
            layer_dict.append(layer_0-2)
        # 输出它本身
        if len(self.layer_indexes) == 0:
            encoder_layer = model.output
        # 分类如果只有一层，取得不正确的话就取倒数第二层
        elif len(self.layer_indexes) == 1:
            if self.layer_indexes[0] in [i + 1 for i in range(len_couche + 1)]:
                encoder_layer = model.get_layer(index=layer_dict[self.layer_indexes[0]]).output
            else:
                encoder_layer = model.get_layer(index=layer_dict[-2]).output
        # 否则遍历需要取的层，把所有层的weight取出来并加起来shape:768*层数
        else:
            # layer_indexes must be [0, 1, 2,3,......24]
            all_layers = [model.get_layer(index=layer_dict[lay]).output
                          if lay in [i + 1 for i in range(len_couche + 1)]
                          else model.get_layer(index=layer_dict[-2]).output  # 如果给出不正确，就默认输出倒数第二层
                          for lay in self.layer_indexes]
            print(self.layer_indexes)
            print(all_layers)
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = Add()(all_layers_select)
            print(encoder_layer.shape)
        output_layer = NonMaskingLayer()(encoder_layer)
        model = Model(model.inputs, output_layer)
        print("load KerasXlnetEmbedding end")
        model.summary(132)


    def xlnet_encode(self, texts):

        # 相当于pool，采用的是https://github.com/terrifyzhao/bert-utils/blob/master/graph.py
        mul_mask = lambda x, m: x * np.expand_dims(m, axis=-1)
        masked_reduce_mean = lambda x, m: np.sum(mul_mask(x, m), axis=1) / (np.sum(m, axis=1, keepdims=True) + 1e-9)

        # 文本预处理
        predicts = []
        for text in texts:
            # print(text)
            tokens = self.tokenizer.encode(text)
            tokens = tokens + [0]*(self.target_len-len(tokens)) if len(tokens) < self.target_len else tokens[0:self.target_len]
            token_input = np.expand_dims(np.array(tokens), axis=0)
            mask_input = np.array([0 if ids == 0 else 1 for ids in tokens])
            segment_input = np.zeros_like(token_input)
            memory_length_input = np.zeros((1, 1))
            # 全局使用，使其可以django、flask、tornado等调用
            with graph.as_default():
                predict = model.predict([token_input, segment_input, memory_length_input], batch_size=1)
                # print(predict)
                prob = predict[0]
                pooled = masked_reduce_mean(prob, [mask_input])
                pooled = pooled.tolist()
                predicts.append(pooled[0])
        return predicts


if __name__ == "__main__":
    xlnet_vector = KerasXlnetVector()
    pooled = xlnet_vector.xlnet_encode(['你是谁呀', '小老弟'])
    print(pooled)
    while True:
        print("input:")
        ques = input()
        print(ques)
        print(xlnet_vector.xlnet_encode([ques]))

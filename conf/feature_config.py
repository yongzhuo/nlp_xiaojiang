# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/5/10 9:13
# @author   :Mo
# @function :path of FeatureProject

import os

# path of BERT model
file_path = os.path.dirname(__file__)
file_path = file_path.replace('conf', '') + 'Data'
model_dir = os.path.join(file_path, 'chinese_L-12_H-768_A-12/')
config_name = os.path.join(model_dir, 'bert_config.json')
ckpt_name = os.path.join(model_dir, 'bert_model.ckpt')
vocab_file = os.path.join(model_dir, 'vocab.txt')
# gpu使用率
gpu_memory_fraction = 0.2
# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]
# 序列的最大程度，单文本建议把该值调小
max_seq_len = 26

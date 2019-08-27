# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/8/27 23:03
# @author   :Mo
# @function :


import pathlib
import sys
import os


# base dir
projectdir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(projectdir)


# path of BERT model
model_dir = os.path.join(projectdir, 'Data', 'chinese_xlnet_mid_L-24_H-768_A-12')
config_name = os.path.join(model_dir, 'xlnet_config.json')
ckpt_name = os.path.join(model_dir, 'xlnet_model.ckpt')
spiece_model = os.path.join(model_dir, 'spiece.model')
attention_type = 'bi' # or 'uni'
# 批处理尺寸
batch_size = 1
# 历史序列长度
memory_len=0
# 当前目标序列长度
target_len=32
# 默认取倒数第二层的输出值作为句向量
layer_indexes = [0, 23] # 可填 0, 1, 2, 3, 4, 5, 6, 7..., 24,其中0为embedding层
# gpu使用率
gpu_memory_fraction = 0.64
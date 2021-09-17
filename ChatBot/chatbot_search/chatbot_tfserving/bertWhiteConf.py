# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/5/13 9:27
# @author  : Mo
# @function: config of Bert-White


import platform
# 适配linux
import sys
import os
# path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
path_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path_root)
print(path_root)


if platform.system().lower() == 'windows':
    # BERT_DIR = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12"
    # BERT_DIR = "D:/soft_install/dataset/bert-model/zuiyi/chinese_roberta_L-4_H-312_A-12_K-104"
    # BERT_DIR = "D:/soft_install/dataset/bert-model/zuiyi/chinese_roberta_L-6_H-384_A-12_K-128"
    BERT_DIR = "D:/soft_install/dataset/bert-model/zuiyi/chinese_simbert_L-4_H-312_A-12"
    # BERT_DIR = "D:/soft_install/dataset/bert-model/zuiyi/chinese_simbert_L-6_H-384_A-12"
else:
    BERT_DIR = "bert/chinese_L-12_H-768_A-12"
    ee = 0

SAVE_DIR = path_root + "/bert_white"
print(SAVE_DIR)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


bert_white_config = {
# 预训练模型路径
"bert_dir": BERT_DIR,
"checkpoint_path": "bert_model.ckpt",  # 预训练模型地址
"config_path": "bert_config.json",
"dict_path": "vocab.txt",
# 预测需要的文件路径
"save_dir": SAVE_DIR,
"path_tfserving": "chatbot_tfserving/1",
"path_docs_encode": "qa.docs.encode.npy",
"path_answers": "qa.answers.json",
"path_qa_idx":  "qa.idx.json",
"path_config": "config.json",
"path_docs": "qa.docs.json",
# 索引构建的存储文件, 如 annoy/faiss
"path_index": "qa.docs.idx",
# 初始语料路径
"path_qa": "chicken_and_gossip.txt",  # QA问答文件地址
# 超参数
"pre_tokenize": None,
"pooling": "cls-1",    # ["first-last-avg", "last-avg", "cls", "pooler", "cls-2", "cls-3", "cls-1"]
"model": "bert",       # bert4keras预训练模型类型
"n_components": 768,   # 降维到 n_components
"n_cluster": 132,      # annoy构建的簇类中心个数n_cluster, 越多效果越好, 计算量就越大
"batch_size": 32,      # 批尺寸
"maxlen": 128,         # 最大文本长度
"ues_white": False,    # 是否使用白化
"use_annoy": False,    # 是否使用annoy
"use_faiss": False,     # 是否使用faiss
"verbose": True,       # 是否显示编码过程日志-batch

"kernel": None,       # bert-white编码后的参数, 可降维
"bias": None,         # bert-white编码后的参数, 偏置bias
"qa_idx": None        # 问题question到答案answer的id对应关系
}

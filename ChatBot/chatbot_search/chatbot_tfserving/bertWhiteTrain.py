# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/4/15 21:59
# @author  : Mo
# @function: encode of bert-whiteing


from __future__ import print_function, division, absolute_import, division, print_function

# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./."))
sys.path.append(path_root)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(path_root)


from bertWhiteTools import txt_read, txt_write, save_json, load_json

from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding
from bert4keras.tokenizers import Tokenizer
from bert4keras.backend import keras, K
from keras.models import Model
import tensorflow as tf

from argparse import Namespace
# from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import time


class NonMaskingLayer(keras.layers.Layer):
    """  去除MASK层
    fix convolutional 1D can"t receive masked input, detail: https://github.com/keras-team/keras/issues/4978
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

    def get_output_shape_for(self, input_shape):
        return input_shape


class BertWhiteModel:
    def __init__(self, config=None):
        """ 初始化超参数、加载预训练模型等 """
        self.config = Namespace(**config)
        self.load_pretrain_model()
        self.eps = 1e-8

    def transform_and_normalize(self, vecs, kernel=None, bias=None):
        """应用变换，然后标准化
        """
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
        norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
        return vecs / np.clip(norms, self.eps, np.inf)

    def compute_kernel_bias(self, vecs):
        """计算kernel和bias
        最后的变换：y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W[:, :self.config.n_components], -mu

    def convert_to_vecs(self, texts):
        """转换文本数据为向量形式
        """

        token_ids = self.convert_to_ids(texts)
        vecs = self.bert_white_encoder.predict(x=[token_ids, np.zeros_like(token_ids)],
                                               batch_size=self.config.batch_size, verbose=self.config.verbose)
        return vecs

    def convert_to_ids(self, texts):
        """转换文本数据为id形式
        """
        token_ids = []
        for text in texts:
            # token_id = self.tokenizer.encode(text, maxlen=self.config.maxlen)[0]
            token_id = self.tokenizer.encode(text, max_length=self.config.maxlen)[0]
            token_ids.append(token_id)
        token_ids = sequence_padding(token_ids)
        return token_ids

    def load_pretrain_model(self):
        """ 加载预训练模型, 和tokenizer """
        self.tokenizer = Tokenizer(os.path.join(self.config.bert_dir, self.config.dict_path), do_lower_case=True)
        # bert-load
        if self.config.pooling == "pooler":
            bert = build_transformer_model(os.path.join(self.config.bert_dir, self.config.config_path),
                                           os.path.join(self.config.bert_dir, self.config.checkpoint_path),
                                           model=self.config.model, with_pool="linear")
        else:
            bert = build_transformer_model(os.path.join(self.config.bert_dir, self.config.config_path),
                                           os.path.join(self.config.bert_dir, self.config.checkpoint_path),
                                           model=self.config.model)
        # output-layers
        outputs, count = [], 0
        while True:
            try:
                output = bert.get_layer("Transformer-%d-FeedForward-Norm" % count).output
                outputs.append(output)
                count += 1
            except:
                break
        # pooling
        if self.config.pooling == "first-last-avg":
            outputs = [NonMaskingLayer()(output_i) for output_i in [outputs[0], outputs[-1]]]
            outputs = [keras.layers.GlobalAveragePooling1D()(fs) for fs in outputs]
            output = keras.layers.Average()(outputs)
        elif self.config.pooling == "first-last-max":
            outputs = [NonMaskingLayer()(output_i) for output_i in [outputs[0], outputs[-1]]]
            outputs = [keras.layers.GlobalMaxPooling1D()(fs) for fs in outputs]
            output = keras.layers.Average()(outputs)
        elif self.config.pooling == "cls-max-avg":
            outputs = [NonMaskingLayer()(output_i) for output_i in [outputs[0], outputs[-1]]]
            outputs_cls = [keras.layers.Lambda(lambda x: x[:, 0])(fs) for fs in outputs]
            outputs_max = [keras.layers.GlobalMaxPooling1D()(fs) for fs in outputs]
            outputs_avg = [keras.layers.GlobalAveragePooling1D()(fs) for fs in outputs]
            output = keras.layers.Concatenate()(outputs_cls + outputs_avg)
        elif self.config.pooling == "last-avg":
            output = keras.layers.GlobalAveragePooling1D()(outputs[-1])
        elif self.config.pooling == "cls-3":
            outputs = [keras.layers.Lambda(lambda x: x[:, 0])(fs) for fs in [outputs[0], outputs[-1], outputs[-2]]]
            output = keras.layers.Concatenate()(outputs)
        elif self.config.pooling == "cls-2":
            outputs = [keras.layers.Lambda(lambda x: x[:, 0])(fs) for fs in [outputs[0], outputs[-1]]]
            output = keras.layers.Concatenate()(outputs)
        elif self.config.pooling == "cls-1":
            output = keras.layers.Lambda(lambda x: x[:, 0])(outputs[-1])
        elif self.config.pooling == "pooler":
            output = bert.output
        # 最后的编码器
        self.bert_white_encoder = Model(bert.inputs, output)
        print("load bert_white_encoder success!" )

    def train(self, texts):
        """
        训练 
        """
        print("读取文本数:".format(len(texts)))
        print(texts[:3])
        # 文本转成向量vecs
        vecs = self.convert_to_vecs(texts)
        # 训练, 计算变换矩阵和偏置项
        self.config.kernel, self.config.bias = self.compute_kernel_bias(vecs)
        if self.config.ues_white:
            # 生成白化后的句子, 即qa对中的q
            vecs = self.transform_and_normalize(vecs, self.config.kernel, self.config.bias)
        return vecs

    def prob(self, texts):
        """
        编码、白化后的向量 
        """
        vecs_encode = self.convert_to_vecs(texts)
        if self.config.ues_white:
            vecs_encode = self.transform_and_normalize(vecs=vecs_encode, kernel=self.config.kernel, bias=self.config.bias)
        return vecs_encode


class BertWhiteFit:
    def __init__(self, config):
        # 训练
        self.bert_white_model = BertWhiteModel(config)
        self.config = Namespace(**config)
        self.docs = []

    def load_bert_white_model(self, path_config):
        """ 模型, 超参数加载 """
        # 超参数加载
        config = load_json(path_config)
        # bert等加载
        self.bert_white_model = BertWhiteModel(config)
        self.config = Namespace(**config)
        # 白化超参数初始化
        self.bert_white_model.config.kernel = np.array(self.bert_white_model.config.kernel)
        self.bert_white_model.config.bias = np.array(self.bert_white_model.config.bias)
        # 加载qa文本数据
        self.answers_dict = load_json(os.path.join(self.config.save_dir, self.config.path_answers))
        self.docs_dict = load_json(os.path.join(self.config.save_dir, self.config.path_docs))
        self.qa_idx = load_json(os.path.join(self.config.save_dir, self.config.path_qa_idx))

        # 加载问题question预训练语言模型bert编码、白化后的encode向量
        self.docs_encode = np.loadtxt(os.path.join(self.config.save_dir, self.config.path_docs_encode))
        # index of vector
        if self.config.use_annoy or self.config.use_faiss:
            from indexAnnoy import AnnoySearch
            self.annoy_model = AnnoySearch(dim=self.config.n_components, n_cluster=self.config.n_cluster)
            self.annoy_model.load(os.path.join(self.config.save_dir, self.config.path_index))
        else:
            self.docs_encode_norm = np.linalg.norm(self.docs_encode, axis=1)
        print("load_bert_white_model success!")

    def read_qa_from_csv(self, sep="\t"):
        """
        从csv文件读取QA对
        """
        # ques_answer = txt_read(os.path.join(self.config.save_dir, self.config.path_qa)) # common qa, sep="\t"
        ques_answer = txt_read(self.config.path_qa)
        self.answers_dict = {}
        self.docs_dict = {}
        self.qa_idx = {}
        count = 0
        for i in range(len(ques_answer)):
            count += 1
            if count > 320:
                break
            ques_answer_sp = ques_answer[i].strip().split(sep)
            if len(ques_answer_sp) != 2:
                print(ques_answer[i])
                continue
            question = ques_answer_sp[0]
            answer = ques_answer_sp[1]
            self.qa_idx[str(i)] = i
            self.docs_dict[str(i)] = question.replace("\n", "").strip()
            self.answers_dict[str(i)] = answer.replace("\n", "").strip()
        self.bert_white_model.config.qa_idx = self.qa_idx

    def build_index(self, vectors):
        """ 构建索引, annoy 或者 faiss """
        if self.config.use_annoy:
            from indexAnnoy import AnnoySearch as IndexSearch
        elif self.config.use_faiss:
            from indexFaiss import FaissSearch as IndexSearch
        self.index_model= IndexSearch(dim=self.config.n_components, n_cluster=self.config.n_cluster)
        self.index_model.fit(vectors)
        self.index_model.save(os.path.join(self.config.save_dir, self.config.path_index))
        print("build index")

    def load_index(self):
        """ 加载索引, annoy 或者 faiss """
        if self.config.use_annoy:
            from indexAnnoy import AnnoySearch as IndexSearch
        elif self.config.use_faiss:
            from indexFaiss import FaissSearch as IndexSearch
        self.index_model = IndexSearch(dim=self.config.n_components, n_cluster=self.config.n_cluster)
        self.index_model.load(self.config.path_index)

    def remove_index(self, ids):
        self.index_model.remove(np.array(ids))

    def predict_with_mmr(self, texts, topk=12):
        """ 维护匹配问题的多样性 """
        from mmr import MMRSum

        res = bwf.predict(texts, topk)
        mmr_model = MMRSum()
        result = []
        for r in res:
            # 维护一个 sim:dict字典存储
            r_dict = {ri.get("sim"):ri for ri in r}
            r_mmr = mmr_model.summarize(text=[ri.get("sim") for ri in r], num=8, alpha=0.6)
            r_dict_mmr = [r_dict[rm[1]] for rm in r_mmr]
            result.append(r_dict_mmr)
        return result

    def predict(self, texts, topk=12):
        """ 预训练模型bert等编码，白化, 获取这一批数据的kernel和bias"""
        texts_encode = self.bert_white_model.prob(texts)
        result = []
        if self.config.use_annoy or self.config.use_faiss:
            index_tops = self.index_model.k_neighbors(vectors=texts_encode, k=topk)
            if self.config.use_annoy:
                for i, index_top in enumerate(index_tops):
                    [dist, idx] = index_top
                    res = []
                    for j, id in enumerate(idx):
                        score = float((2 - (dist[j] ** 2)) / 2)
                        res_i = {"score": score, "text": texts[i], "sim": self.docs_dict[str(id)],
                                 "answer": self.answers_dict[str(id)]}
                        res.append(res_i)
                    result.append(res)
            else:
                distances, indexs = index_tops
                for i in range(len(distances)):
                    res = []
                    for j in range(len(distances[i])):
                        score = distances[i][j]
                        id = indexs[i][j]
                        id = id if id != -1 else len(self.docs_dict) - 1
                        res_i = {"score": score, "text": texts[i], "sim": self.docs_dict[str(id)],
                                 "answer": self.answers_dict[str(id)]}
                        res.append(res_i)
                    result.append(res)
        else:
            for i, te in enumerate(texts_encode):
                # scores = np.matmul(texts_encode, self.docs_encode_reshape)
                facot_1 = te * self.docs_encode
                te_norm = np.linalg.norm(te)
                facot_2 = te_norm * self.docs_encode_norm
                score = np.sum(facot_1, axis=1) / (facot_2 + 1e-9)
                idxs = np.argsort(score)[::-1]
                res = []
                for j in idxs[:topk]:
                    res_i = {"score": float(score[j]), "text": texts[i], "sim": self.docs_dict[str(j)],
                             "answer": self.answers_dict[str(j)]}
                    res.append(res_i)
                result.append(res)
        return result

    def trainer(self):
        """ 预训练模型bert等编码，白化, 获取这一批数据的kernel和bias """
        # 加载数据
        self.read_qa_from_csv()
        # bert编码、训练
        self.docs_encode = self.bert_white_model.train([self.docs_dict.get(str(i), "") for i in range(len(self.docs_dict))])
        self.bert_white_model.config.kernel = self.bert_white_model.config.kernel.tolist()
        self.bert_white_model.config.bias = self.bert_white_model.config.bias.tolist()
        # 存储qa文本数据
        save_json(self.bert_white_model.config.qa_idx, os.path.join(self.config.save_dir, self.config.path_qa_idx))
        save_json(self.answers_dict, os.path.join(self.config.save_dir, self.config.path_answers))
        save_json(self.docs_dict, os.path.join(self.config.save_dir, self.config.path_docs))
        # 包括超参数等
        save_json(vars(self.bert_white_model.config), os.path.join(self.config.save_dir, self.config.path_config))
        # 存储问题question预训练语言模型bert编码、白化后的encode向量
        np.savetxt(os.path.join(self.config.save_dir, self.config.path_docs_encode), self.docs_encode)
        # 索引 或者 正则化
        if self.config.use_annoy or self.config.use_faiss:
            self.build_index(self.docs_encode.astype(np.float32))
        else:
            self.docs_encode_norm = np.linalg.norm(self.docs_encode, axis=1)
        print(" bert-white-trainer success! ")



if __name__ == '__main__':
    # 训练并存储
    from bertWhiteConf import bert_white_config
    bwf = BertWhiteFit(config=bert_white_config)
    bwf.trainer()

    texts = ["小姜机器人", "你叫什么名字"]
    res = bwf.predict(texts)
    print(res)
    res_mmr = bwf.predict_with_mmr(texts)
    print(res_mmr)

    # bwf.index_model.remove([i for i in range(100)])

    while True:
        print("请输入:")
        ques = input()
        res_mmr = bwf.predict_with_mmr(texts)
        print(res_mmr)
        res = bwf.predict([ques])
        print(res)



# python bertWhiteTrain.py




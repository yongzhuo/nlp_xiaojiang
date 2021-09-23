# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/9 23:05
# @author   :Mo
# @function :SimBERT再训练BERT-base(NSP任务), UNILM的生成能力
# @reference:https://github.com/ZhuiyiTechnology/roformer-sim
# 目前仅保证支持 Tensorflow 1.x + Keras <= 2.3.1 + bert4keras>=0.10.6。
# 具体用法请看 https://github.com/bojone/bert4keras/blob/8ffb46a16a79f87aa8cdf045df7994036b4be47d/bert4keras/snippets.py#L580


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.backend import keras, K
import numpy as np
import time


# bert配置
# BERT_DIR = "D:/soft_install/dataset/bert-model/zuiyi/chinese_roformer-sim-char_L-12_H-768_A-12"
BERT_DIR = "D:/soft_install/dataset/bert-model/zuiyi/chinese_roformer-sim-char_L-6_H-384_A-6"

config_path = BERT_DIR + "/bert_config.json"
checkpoint_path = BERT_DIR + "/bert_model.ckpt"
dict_path = BERT_DIR + "/vocab.txt"
maxlen = 128


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    model='roformer',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(seq2seq).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95, mask_idxs=[]):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        for i in mask_idxs:
            token_ids[i] = tokenizer._token_mask_id
        output_ids = self.random_sample([token_ids, segment_ids], n, topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen)


def gen_synonyms(text, n=100, k=20):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    """
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


if __name__ == '__main__':
    datas = [{"text": "平乐县，古称昭州，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭城，总面积1919.34平方公里。"},
             {"text": "平乐县主要旅游景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐县为漓江分界点，平乐以北称漓江，以南称桂江，是著名的大桂林旅游区之一。"},
             {"text": "印岭玲珑，昭水晶莹，环绕我平中。青年的乐园，多士受陶熔。生活自觉自治，学习自发自动。五育并重，手脑并用。迎接新潮流，建设新平中"},
             {"text": "桂林山水甲天下, 阳朔山水甲桂林"},
             {"text": "三国一统天下"},
             {"text": "世间万物皆系于其上"},
             {"text": "2020年真是一个糟糕的年代, 进入20年代，新冠爆发、经济下行，什么的都来了。"},
             {"text": "仿佛一切都变得不那么重要了。"},
             {"text": "苹果多少钱一斤"}
             ]
    time_start = time.time()
    for da in datas:
        text = da.get("text", "")
        res = gen_synonyms(text)
        print(res)
    time_total = time.time()-time_start
    print("time_total:{}".format(time_total))
    print("time_per:{}".format(time_total/len(datas)/20))

    while True:
        print("请输入:")
        text = input()
        res = gen_synonyms(text)
        print(res)

['平乐县古称昭州，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，西毗邻阳朔，北连恭城',
 '平乐县，古称昭州，隶属于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭城，总面积1919.',
 '平乐县隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭',
 '平乐县，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连',
 '平乐县，属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭',
 '平乐县，古称昭州，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平',
 '平乐县，古称昭州，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平县。',
 '平乐县，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔',
 '平乐县，古称昭州，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭城，总面积1919.34',
 '平乐县，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔。',
 '平乐县，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平县，西北毗邻阳朔县。',
 '平乐县隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔',
 '平乐县隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔。',
 '昭平县，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连',
 '昭乐县隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔。',
 '昭州，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭',
 '平乐县，古称昭州，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县',
 '广西壮族自治区桂林市，隶属于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭城，总面积1919',
 '平乐县，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平',
 '广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭城，总面积1919']

['榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐县为漓江分界点，平乐以北称漓江。',
 '景点主要就是榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐县为漓江分界点',
 '主要景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐以北称漓江，以南称桂',
 '景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐以北称漓江，以南称桂江',
 '榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐县为漓江分界点',
 '榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐县为漓江分界点。',
 '这个景点主要有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐以北称漓江，以南称',
 '1. 榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐县为漓江分界点。',
 '第一，平乐县的主要旅游景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等。',
 '平乐县主要旅游景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等。',
 '学校主要旅游景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐以北称漓江。',
 '1、榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐以北称漓江，以南称桂江。',
 '宜宾市榕津千年古榕、冷水石景苑、仙家温泉、漓江风景区等，普洱山为漓江分界点，平乐以北称漓江。',
 '花开湖畔的景观特色：榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐以北称漓江',
 '广州市榕津千年古榕、冷水石景苑、仙家温泉、漓江风景区、漓江风景区等，平乐县为漓江分界点。',
 '该景区主要旅游景点有：榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等。',
 '郴州市永化区榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，桂江县为漓江分界点，平',
 '公益群岛水平县主要旅游景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等',
 '此处主要是桂江最为优秀的桂江旅游区，主要是榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区等。',
 '桂江“大桂林”景点”的三角半南边有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等。']

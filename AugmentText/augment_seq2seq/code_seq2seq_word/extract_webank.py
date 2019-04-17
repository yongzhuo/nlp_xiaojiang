"""
把文件格式转换为可训练格式
Code from: QHDuan(2018-02-05) url: https://github.com/qhduan/just_another_seq2seq
"""

import re
import sys
import pickle
import jieba
import gensim
import numpy as np
from tqdm import tqdm
from conf.path_config import projectdir
from conf.path_config import w2v_model_merge_short_path
from utils.mode_util.seq2seq.word_sequence import WordSequence

from conf.path_config import model_ckpt_web_anti_word
from conf.path_config import train_data_web_xyw_anti
from conf.path_config import train_data_web_emb_anti
from conf.path_config import path_webank_sim

sys.path.append('..')


def make_split(line):
    """构造合并两个句子之间的符号
    """
    if re.match(r'.*([，。…？！～\.,!?])$', ''.join(line)):
        return []
    return ['，']


def good_line(line):
    """判断一个句子是否好"""
    if len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) > 2:
        return False
    return True


def regular(sen, limit=50):
    sen = re.sub(r'\.{3,100}', '…', sen)
    sen = re.sub(r'…{2,100}', '…', sen)
    sen = re.sub(r'[,]{1,100}', '，', sen)
    sen = re.sub(r'[\.]{1,100}', '。', sen)
    sen = re.sub(r'[\?]{1,100}', '？', sen)
    sen = re.sub(r'[!]{1,100}', '！', sen)
    if len(sen) > limit:
        sen = sen[0:limit]
    return sen


def creat_train_data_of_bank_corpus(limit=50, x_limit=3, y_limit=3):
    """执行程序
    Args:
        limit: 只输出句子长度小于limit的句子
    """

    print('load word2vec start!')
    word_vec = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_merge_short_path, encoding='gbk', binary=False, limit=None)
    print('load word2vec end!')
    fp = open(path_webank_sim, 'r', encoding='gbk', errors='ignore')

    x_datas = []
    y_datas = []
    max_len = 0
    count_fp = 0
    for line in tqdm(fp):
        count_fp += 1
        if count_fp == 1:
            continue
        sim_bank_datas_one_split = line.strip().split(",")
        len_x1 = len(sim_bank_datas_one_split[0])
        len_x2 = len(sim_bank_datas_one_split[1])
        # if max_len < len_x1 or max_len < len_x2:
        max_len = max(len_x1, len_x2, max_len)

        sentence_org = regular(sim_bank_datas_one_split[0], limit=limit)
        sentence_sim = regular(sim_bank_datas_one_split[1], limit=limit)
        org_cut = jieba._lcut(sentence_org)
        sen_cut = jieba._lcut(sentence_sim)

        x_datas.append(org_cut)
        y_datas.append(sen_cut)
        x_datas.append(sen_cut)
        y_datas.append(org_cut)

    print(len(x_datas), len(y_datas))
    for ask, answer in zip(x_datas[:50], y_datas[:50]):
        print(''.join(ask))
        print(''.join(answer))
        print('-' * 50)

    data = list(zip(x_datas, y_datas))
    data = [
        (x, y)
        for x, y in data
        if len(x) < limit \
        and len(y) < limit \
        and len(y) >= y_limit \
        and len(x) >= x_limit
    ]
    x_data, y_data = zip(*data)

    print('refine train data')

    train_data = x_data + y_data

    print('fit word_sequence')

    ws_input = WordSequence()

    ws_input.fit(train_data, max_features=100000)

    print('dump word_sequence')

    pickle.dump((x_data, y_data, ws_input),
        open(train_data_web_xyw_anti, 'wb')
    )

    print('make embedding vecs')

    emb = np.zeros((len(ws_input), len(word_vec['</s>'])))

    np.random.seed(1)
    for word, ind in ws_input.dict.items():
        if word in word_vec:
            emb[ind] = word_vec[word]
        else:
            emb[ind] = np.random.random(size=(300,)) - 0.5

    print('dump emb')

    pickle.dump(
        emb,
        open(train_data_web_emb_anti, 'wb')
    )

    print('done')


if __name__ == '__main__':
    creat_train_data_of_bank_corpus()
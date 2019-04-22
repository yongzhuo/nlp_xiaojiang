"""把 chicken_and_gossip数据 文件格式转换为可训练格式
Code from: QHDuan(2018-02-05) url: https://github.com/qhduan/just_another_seq2seq
"""

import re
import sys
import pickle
import jieba
import gensim
import numpy as np
from tqdm import tqdm
from utils.text_tools import txtRead
from conf.path_config import word2_vec_path
from conf.path_config import chicken_and_gossip_path
from conf.path_config import w2v_model_merge_short_path
from utils.mode_util.seq2seq.word_sequence import WordSequence

from conf.path_config import chatbot_data_cg_xyw_anti_word
from conf.path_config import chatbot_data_cg_emb_anti_word
from conf.path_config import model_ckpt_cg_anti_word


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


def creat_train_data_of_cg_corpus(limit=50, x_limit=2, y_limit=2):

    print('load word2vec start!')
    word_vec_short = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_merge_short_path, binary=False, limit=None, encoding='gbk')
    print('load word_vec_short start!')
    word_vec = gensim.models.KeyedVectors.load_word2vec_format(word2_vec_path, binary=False, limit=None)
    print('load word_vec end!')

    x_datas = []
    y_datas = []
    max_len = 0
    sim_ali_web_gov_dli_datas = txtRead(chicken_and_gossip_path, encodeType="utf-8")
    for sim_ali_web_gov_dli_datas_one in sim_ali_web_gov_dli_datas[1:]:
        if sim_ali_web_gov_dli_datas_one:
            sim_ali_web_gov_dli_datas_one_split = sim_ali_web_gov_dli_datas_one.strip().split("\t")
            if len(sim_ali_web_gov_dli_datas_one_split) == 2:
                # if sim_ali_web_gov_dli_datas_one_split[2]=="1":
                len_x1 = len(sim_ali_web_gov_dli_datas_one_split[0])
                len_x2 = len(sim_ali_web_gov_dli_datas_one_split[1])
                # if max_len < len_x1 or max_len < len_x2:
                max_len = max(len_x1, len_x2, max_len)

                sentence_org = regular(sim_ali_web_gov_dli_datas_one_split[0], limit=limit)
                sentence_sim = regular(sim_ali_web_gov_dli_datas_one_split[1], limit=limit)
                org_cut = jieba._lcut(sentence_org)
                sen_cut = jieba._lcut(sentence_sim)

                x_datas.append(org_cut)
                y_datas.append(sen_cut)

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
                open(chatbot_data_cg_xyw_anti_word, 'wb')
                )

    print('make embedding vecs')

    emb = np.zeros((len(ws_input), len(word_vec_short['</s>'])))

    np.random.seed(1)
    for word, ind in ws_input.dict.items():
        if word in word_vec:
            emb[ind] = word_vec[word]
        else:
            emb[ind] = np.random.random(size=(300,)) - 0.5

    print('dump emb')

    pickle.dump(
        emb,
        open(chatbot_data_cg_emb_anti_word, 'wb')
    )

    print('done')


if __name__ == '__main__':
    creat_train_data_of_cg_corpus()
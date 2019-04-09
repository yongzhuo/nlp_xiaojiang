# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/4 10:00
# @author   :Mo
# @function :chatbot based search, encode sentence_vec by char

from conf.path_config import w2v_model_char_path
from conf.path_config import matrix_ques_part_path_char
from utils.text_tools import txtRead, txtWrite, getChinese
from conf.path_config import projectdir, chicken_and_gossip_path
from numpy import float32 as numpy_type
from collections import Counter
import pickle, jieba, os, re
import jieba.posseg as pseg
from gensim import matutils
from math import log
import numpy as np
import gensim
import jieba


def load_word2vec_model(path, bin=False, limit=None):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(path, limit=limit, binary=bin, unicode_errors='ignore')
    return word2vec_model


def encoding_question(w2v_model, char_list):
    '''    生成句子向量
    :param wordlist: 分词list
    :param is_replaced: 是否替换default true
    :param debug_mode: default false
    :return: array句子的向量 len=300
    '''
    try:
        sentence_vec = w2v_model.wv[word2vec_model.index2word[1]] * 0
    except:
        sentence_vec = w2v_model.wv[word2vec_model.index2word[0]] * 0

    for k in range(len(char_list)):
        char_list_one = char_list[k]
        if type(char_list_one) == str:
            try:
                sentence_vec = sentence_vec + w2v_model.wv[char_list_one]
            except Exception as e:
                print(str(e))
                if char_list_one not in [' ', '']:
                    sentence_vec = sentence_vec + 1
    return sentence_vec


def most_similar_sentence_vec(vec_ques, matrix_org, top_vec=20):
    """
      最相似的句子，句向量与矩阵点乘
    :param vec: 
    :param matrix: 
    :param keys: 
    :param topn: 
    :return: 
    """
    # 首先对句向量矩阵标号
    matrix_org_index = list(range(len(matrix_org)))
    # Scale a vector to unit length. The only exception is the zerovector, which is returned back unchanged.
    vec_ques_mean = matutils.unitvec(np.array([vec_ques]).mean(axis=0)).astype(numpy_type)
    # matrix_org单位化
    matrix_org_norm = (matrix_org / np.sqrt((matrix_org ** 2).sum(-1))[..., np.newaxis]).astype(numpy_type)
    # 计算两个向量之间的相似度，使用numpy的dot函数，矩阵点乘
    matrix_vec_dot = np.dot(matrix_org_norm, vec_ques_mean)
    # 防止top_vec越界
    top_vec = min(len(matrix_org), top_vec)
    # 相似度排序
    most_similar_sentence_vec_sort = matutils.argsort(matrix_vec_dot, topn=top_vec, reverse=True)

    index_score = []
    for t in most_similar_sentence_vec_sort[:top_vec]:
        index_score.append([matrix_org_index[t], float(matrix_vec_dot[t])])
    return index_score


def create_matrix_org_np(sen_count, word2vec_model, qa_path, matrix_ques_path):
    """
      创建问题句向量
    :param sen_count: int
    :param word2vec_model: gensim model
    :param qa_path: str
    :param matrix_ques_path:str 
    :return: None
    """
    if os.path.exists(matrix_ques_path):
        file_matrix_ques = open(matrix_ques_path, 'rb')
        matrix_ques = pickle.load(file_matrix_ques)
        return matrix_ques
    print('create_matrix_org_pkl start!')
    qa_dail = txtRead(qa_path, encodeType='utf-8')
    # questions = []
    matrix_ques = []
    count = 0
    for qa_dail_one in qa_dail:
        ques = getChinese(qa_dail_one.split('\t')[0])
        char_list = [ques_char for ques_char in ques]
        sentence_vec = encoding_question(word2vec_model, char_list)
        matrix_ques.append(sentence_vec)
        if len(matrix_ques)%sen_count == 0 and len(matrix_ques) != 0:
            print("count: " + str(count))
            count += 1
            np.savetxt(projectdir + "/Data/sentence_vec_encode_char/" + str(count)+".txt", matrix_ques)
            matrix_ques = []
            break

    # count += 1
    # np.savetxt(projectdir + "/Data/sentence_vec_encode_char/" + str(count)+".txt", matrix_ques)

    print('create_matrix_org_pkl ok!')
    # return matrix_ques


if __name__ == '__main__':

    # 读取问答语料
    syn_qa_dails = txtRead(chicken_and_gossip_path, encodeType='utf-8')
    # 读取词向量
    word2vec_model = load_word2vec_model(w2v_model_char_path, limit=None)
    # 创建标准问答中问题的句向量，存起来，到matrix_ques_path， 10万条，可自己设置，这里需要耗费点时间
    if not os.path.exists(matrix_ques_part_path_char):
        # matrix_ques = create_matrix_org_np(sen_count=100000, word2vec_model=word2vec_model, qa_path=chicken_and_gossip_path, matrix_ques_path=matrix_ques_part_path_char)
        create_matrix_org_np(sen_count=100000, word2vec_model=word2vec_model, qa_path=chicken_and_gossip_path, matrix_ques_path=matrix_ques_part_path_char)
    # 重载
    matrix_ques = np.loadtxt(matrix_ques_part_path_char)
    print("np.loadtxt(matrix_ques_part_path_char) ok!")
    while True:
        print("你问: ")
        ques_ask = input()
        ques_clean = getChinese(ques_ask)
        char_list = [ques_char for ques_char in ques_clean]
        sentence_vic = encoding_question(word2vec_model, char_list)
        top_20_qid = most_similar_sentence_vec(sentence_vic, matrix_ques, top_vec=20)
        try:
            print("小姜机器人: " + syn_qa_dails[top_20_qid[0][0]].strip().split("\t")[1])
            print([(syn_qa_dails[top_20_qid[i][0]].strip().split("\t")[0], syn_qa_dails[top_20_qid[i][0]].strip().split("\t")[1]) for i in range(len(top_20_qid))])
        except Exception as e:
            # 有的字符可能打不出来
            print(str(e))


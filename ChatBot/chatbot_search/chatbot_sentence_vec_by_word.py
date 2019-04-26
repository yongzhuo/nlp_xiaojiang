# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/4 10:00
# @author   :Mo
# @function :chatbot based search, encode sentence_vec by word


from conf.path_config import w2v_model_merge_short_path, w2v_model_wiki_word_path
from conf.path_config import projectdir, chicken_and_gossip_path
from utils.text_tools import txtRead, txtWrite, getChinese
from conf.path_config import matrix_ques_part_path
from numpy import float32 as numpy_type
from collections import Counter
import pickle, jieba, os, re
import jieba.posseg as pseg
from gensim import matutils
from math import log
import numpy as np
import gensim
import jieba
import time


def load_word2vec_model(path, bin=False, limit=None):
    print("load_word2vec_model start!")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(path, limit=limit, binary=bin, unicode_errors='ignore')
    print("load_word2vec_model end!")
    return word2vec_model


def get_td_idf_flag(jieba_cut_list, dictionary, tfidf_model):
    # todo
    '''获取td-idf权重，有问题，同一个词只计算一次，有的还没有，比如说停用词'''
    seg1_list = []
    vec1 = tfidf_model[dictionary.doc2bow(jieba_cut_list)]
    for vec1_one in vec1:
        seg1_list.append(vec1_one[1])
    sum_seg1_list = sum(seg1_list)

    return [x/sum_seg1_list for x in seg1_list]


def get_jieba_flag(flag):
    '''词性'''
    if flag in ['n', 'nr', 'ns', 'nt', 'nz']:
        weight = 1.3
    elif flag in ['r', 'i', 't', 'ng', 'an']:
        weight = 0.7
    else:
        weight = 1
    return weight


def word_segment_process(sentence):
    """
        jieba切词\词性
    :param sentence: 
    :return: 
    """
    sentence = sentence.replace('\n', '').replace(',', '').replace('"', '').replace(' ', '').replace('\t', '').upper().strip()
    word_list = []
    flag_list = []
    try:
        sentence_cut =  ''.join(jieba.lcut(sentence, cut_all=False, HMM=False))
        words = pseg.cut(sentence_cut)
        for word in words:
            word_list.append(word.word)
            flag_list.append(word.flag)
    except Exception as e:
        word_list = [sentence]
        flag_list = ['nt']
    return word_list, flag_list


def encoding_question(w2v_model, word_list, flag_list):
    '''    生成句子向量
    :param wordlist: 分词list
    :param is_replaced: 是否替换default true
    :param debug_mode: default false
    :return: array句子的向量 len=300
    '''
    try:
        sentence_vec = w2v_model.wv[w2v_model.index2word[1]] * 0
    except:
        sentence_vec = w2v_model.wv[w2v_model.index2word[1]] * 0

    for k in range(len(word_list)):
        word = word_list[k]
        flag = flag_list[k]
        if type(word) == str:
            try:
                sentence_vec = sentence_vec + w2v_model.wv[word] * get_jieba_flag(flag)
            except Exception as e:
                if word not in [' ', '']:
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


def create_matrix_org_np(sen_count, word2vec_model, qa_path, matrix_ques_path_word):
    """
      创建问题句向量,设置sen_count=10000, 防止内存不够奔溃
    :param sen_count: int, write sentence_encode num per twice
    :param word2vec_model: model
    :param qa_path: str
    :param matrix_ques_path: str
    :return: 
    """
    if os.path.exists(matrix_ques_path_word):
        file_matrix_ques = open(matrix_ques_path_word, 'rb')
        matrix_ques = pickle.load(file_matrix_ques)
        return matrix_ques
    print('create_matrix_org_pkl start!')
    qa_dail = txtRead(qa_path, encodeType='utf-8')
    # questions = []
    matrix_ques = []
    count = 0
    for qa_dail_one in qa_dail:
        ques = getChinese(qa_dail_one.split('\t')[0])
        # questions.append(ques)
        word_list, flag_list = word_segment_process(ques)
        sentence_vec = encoding_question(word2vec_model, word_list, flag_list)
        matrix_ques.append(sentence_vec)
        if len(matrix_ques)%sen_count == 0 and len(matrix_ques) != 0:
            print("count: " + str(count))
            count += 1
            np.savetxt(projectdir + "/Data/sentence_vec_encode_word/" + str(count)+".txt", matrix_ques)
            matrix_ques = []
            # break

    count += 1
    np.savetxt(projectdir + "/Data/sentence_vec_encode_word/" + str(count)+".txt", matrix_ques)
    # matrix_ques = []
    # file_matrix_ques = open(matrix_ques_path, 'wb')
    # pickle.dump(matrix_ques, file_matrix_ques)
    print('create_matrix_org_np ok!')
    # return matrix_ques


if __name__ == '__main__':
    # 读取问答语料
    syn_qa_dails = txtRead(chicken_and_gossip_path, encodeType='utf-8')

    # 读取词向量，w2v_model_wiki_word_path数据是自己训练的，w2v_model_merge_short_path只取了部分数据，你可以前往下载
    if os.path.exists(w2v_model_wiki_word_path):
        word2vec_model = load_word2vec_model(w2v_model_wiki_word_path, limit=None)
        print("load w2v_model_wiki_word_path ok!")
    else:
        word2vec_model = load_word2vec_model(w2v_model_merge_short_path, limit=None)
        print("load w2v_model_merge_short_path ok!")

    # 创建标准问答中问题的句向量，存起来，到matrix_ques_path
    if not os.path.exists(matrix_ques_part_path):
        create_matrix_org_np(sen_count=100000, word2vec_model=word2vec_model, qa_path=chicken_and_gossip_path, matrix_ques_path_word=matrix_ques_part_path)

    # 读取
    print("np.loadtxt(matrix_ques_part_path) start!")
    matrix_ques = np.loadtxt(matrix_ques_part_path)
    print("np.loadtxt(matrix_ques_part_path) end!")
    while True:
        print("你: ")
        ques_ask = input()
        ques_clean = getChinese(ques_ask)
        word_list, flag_list = word_segment_process(ques_clean)
        sentence_vic = encoding_question(word2vec_model, word_list, flag_list)
        top_20_qid = most_similar_sentence_vec(sentence_vic, matrix_ques, top_vec=20)
        try:
            print("小姜机器人: " + syn_qa_dails[top_20_qid[0][0]].strip().split("\t")[1])
            print([(syn_qa_dails[top_20_qid[i][0]].strip().split("\t")[0], syn_qa_dails[top_20_qid[i][0]].strip().split("\t")[1]) for i in range(len(top_20_qid))])
        except Exception as e:
            # 有的字符可能打不出来
            print(str(e))

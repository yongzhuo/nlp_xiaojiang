# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/4 10:00
# @author   :Mo
# @function :calculate distances of Varity

from sklearn.feature_extraction.text import TfidfVectorizer
from utils.text_tools import txtRead, get_syboml, strQ2B
import Levenshtein as Leven
from fuzzywuzzy import fuzz
import jieba.analyse
import numpy as np
import xpinyin
import pickle
import jieba
import os


zero_bit = 0.000000001
pin = xpinyin.Pinyin()


def clear_sentence(sentence):
    """
      数据清晰，全角转半角
    :param sentence: str, input sentence
    :return: str, clearned sentences
    """
    corpus_one_clear = str(sentence).replace(' ', '').strip()
    ques_q2b = strQ2B(corpus_one_clear.strip())
    ques_q2b_syboml = get_syboml(ques_q2b)
    return ques_q2b_syboml


def chinese2pinyin(sentence):
    """
      chinese translate to pingyin
    :param sentence: str, input sentence
    :return: str, output pingyin
    """
    ques_q2b_syboml_pinying = pin.get_pinyin(sentence, ' ')
    return ques_q2b_syboml_pinying


def hamming_distance(v1, v2):
    n = int(v1, 2) ^ int(v2, 2)
    return bin(n & 0xffffffff).count('1')


def cosine_distance(v1, v2): # 余弦距离
    if v1.all() and v2.all():
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    else:
        return 0


def euclidean_distance(v1, v2):  # 欧氏距离
    return np.sqrt(np.sum(np.square(v1 - v2)))


def manhattan_distance(v1, v2):  # 曼哈顿距离
    return np.sum(np.abs(v1 - v2))


def chebyshev_distance(v1, v2):  # 切比雪夫距离
    return np.max(np.abs(v1 - v2))


def minkowski_distance(v1, v2):  # 闵可夫斯基距离
    return np.sqrt(np.sum(np.square(v1 - v2)))


def euclidean_distance_standardized(v1, v2):  # 标准化欧氏距离
    v1_v2 = np.vstack([v1, v2])
    sk_v1_v2 = np.var(v1_v2, axis=0, ddof=1)
    return np.sqrt(((v1 - v2) ** 2 / (sk_v1_v2 + zero_bit * np.ones_like(sk_v1_v2))).sum())


def mahalanobis_distance(v1, v2):  # 马氏距离
    # 马氏距离要求样本数要大于维数，否则无法求协方差矩阵
    # 此处进行转置，表示10个样本，每个样本2维
    X = np.vstack([v1, v2])
    XT = X.T

    # 方法一：根据公式求解
    S = np.cov(X)  # 两个维度之间协方差矩阵
    try:
        SI = np.linalg.inv(S)  # 协方差矩阵的逆矩阵  todo
    except:
        SI = np.zeros_like(S)
    # 马氏距离计算两个样本之间的距离，此处共有10个样本，两两组合，共有45个距离。
    n = XT.shape[0]
    distance_all = []
    for i in range(0, n):
        for j in range(i + 1, n):
            delta = XT[i] - XT[j]
            distance_1 = np.sqrt(np.dot(np.dot(delta, SI), delta.T))
            distance_all.append(distance_1)
    return np.sum(np.abs(distance_all))


def bray_curtis_distance(v1, v2):  # 布雷柯蒂斯距离, 生物学生态距离
    up_v1_v2 = np.sum(np.abs(v2 - v1))
    down_v1_v2 = np.sum(v1) + np.sum(v2)
    return up_v1_v2 / (down_v1_v2 + zero_bit)


def pearson_correlation_distance(v1, v2):  # 皮尔逊相关系数（Pearson correlation）
    v1_v2 = np.vstack([v1, v2])
    return np.corrcoef(v1_v2)[0][1]


def jaccard_similarity_coefficient_distance(v1, v2):  # 杰卡德相似系数(Jaccard similarity coefficient)
    # 方法一：根据公式求解
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    up = np.double(np.bitwise_and((v1 != v2), np.bitwise_or(v1 != 0, v2 != 0)).sum())
    down = np.double(np.bitwise_or(v1 != 0, v2 != 0).sum() + zero_bit)
    return up / down


def wmd_distance(model, sent1_cut_list, sent2_cut_list):  # WMD距离
    # model.init_sims(replace=True)
    distance = model.wmdistance(sent1_cut_list, sent2_cut_list)
    return distance


# def HamMings_Levenshtein(str1, str2):
#     sim = Leven.hamming(str1, str2)
#     return sim

def edit_levenshtein(str1, str2):
    return Leven.distance(str1, str2)


def ratio_levenshtein(str1, str2):
    return Leven.ratio(str1, str2)


def jaro_levenshtein(str1, str2):
    return Leven.jaro(str1, str2)


def set_ratio_fuzzywuzzy(str1, str2):
    return fuzz.token_set_ratio(str1, str2)


def sort_ratio_fuzzywuzzy(str1, str2):
    return fuzz.token_sort_ratio(str1, str2)


def num_of_common_sub_str(str1, str2):
    '''
    求两个字符串的最长公共子串
    思想：建立一个二维数组，保存连续位相同与否的状态
    '''
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    maxNum = 0  # 最长匹配长度
    p = 0  # 匹配的起始位

    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i + 1][j + 1]
                    # 记录最大匹配长度的终止位置
                    p = i + 1
    # return str1[p - maxNum:p], maxNum
    return maxNum


#######################################################  汉明距离
def string_hash(source):
    if source == "":
        return 0
    else:
        x = ord(source[0]) << 7
        m = 1000003
        mask = 2 ** 128 - 1
        for c in source:
            x = ((x * m) ^ ord(c)) & mask
        x ^= len(source)
        if x == -1:
            x = -2
        x = bin(x).replace('0b', '').zfill(64)[-64:]

        return str(x)


def sim_hash(content):
    seg = jieba.cut(content)
    keyWord = jieba.analyse.extract_tags('|'.join(seg), topK=20, withWeight=True, allowPOS=())
    # 先按照权重排序，再按照词排序
    keyList = []
    # print(keyWord)
    for feature, weight in keyWord:
        weight = int(weight * 20)
        feature = string_hash(feature)
        temp = []
        for f in feature:
            if f == '1':
                temp.append(weight)
            else:
                temp.append(-weight)
        keyList.append(temp)
    content_list = np.sum(np.array(keyList), axis=0)
    # 编码读不出来
    if len(keyList) == 0:
        return '00'
    simhash = ''
    for c in content_list:
        if c > 0:
            simhash = simhash + '1'
        else:
            simhash = simhash + '0'
    return simhash


def hamming_distance_equal(v1, v2):
    n = int(v1, 2) ^ int(v2, 2)
    return bin(n & 0xffffffff).count('1')


def hamming_distance(sen1, sen2):
    return hamming_distance_equal(sim_hash(sen1), sim_hash(sen2))


def normalization(x):
    """
      归一化，最大最小值
    :param x: 
    :return:  
    """
    return [(float(i) - min(x)) / float(max(x) - min(x) + zero_bit) for i in x]


def z_score(x, axis=0):
    """
      标准化
    :param x: arrary, numpy
    :param axis: int, 0
    :return: arrary, numpy
    """
    x = np.array(x).astype(float)
    xr = np.rollaxis(x, axis=axis)
    xr -= np.mean(x, axis=axis)
    xr /= np.std(x, axis=axis)
    # print(x)
    return x


def tok_td_idf(data_path):
    if os.path.exists(data_path + 'td_idf_cut.csv'):
        '''#计算TD-DIDF，获取训练测试数据'''
        datas = txtRead(data_path + 'td_idf_cut.csv')
        # 默认值只匹配长度≥2的单词,修改为1；ngram_range特征所以有2个词的,总计词语50428个
        # vec_tdidf = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", min_df=1, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1,max_features=30000)
        vec_tdidf = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", min_df=3,
                                    max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1, max_features=50000)
        vec_tdidf.fit_transform(datas)
        file_vec_tdidf = open(data_path + 'td_idf_cut_model.pkl', 'wb')
        pickle.dump(vec_tdidf, file_vec_tdidf)

    return vec_tdidf


def tok_td_idf_pinyin(data_path):
    if os.path.exists(data_path + 'td_idf_cut_pinyin.csv'):
        '''#计算TD-DIDF，获取训练测试数据'''
        datas = txtRead(data_path + 'td_idf_cut_pinyin.csv')
        # 默认值只匹配长度≥2的单词,修改为1；ngram_range特征所以有2个词的,总计词语50428个
        # vec_tdidf = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", min_df=1, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1,max_features=30000)
        vec_tdidf = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", min_df=3,
                                    max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1, max_features=50000)
        vec_tdidf.fit_transform(datas)
        file_vec_tdidf = open(data_path + 'td_idf_cut_pinyin_model.pkl', 'wb')
        pickle.dump(vec_tdidf, file_vec_tdidf)

    return vec_tdidf


if __name__ == '__main__':
    vec1_test = np.array([1, 38, 17, 32])
    vec2_test = np.array([5, 6, 8, 9])

    str1_test = "你到底是谁?"
    str2_test = "没想到我是谁，是真样子"

    print(clear_sentence(str1_test))  # 数据处理
    print(chinese2pinyin(str1_test))  # 中文转拼音

    print(euclidean_distance(vec1_test, vec2_test))
    print(cosine_distance(vec1_test, vec2_test))
    print(manhattan_distance(vec1_test, vec2_test))
    print(euclidean_distance(vec1_test, vec2_test))
    print(chebyshev_distance(vec1_test, vec2_test))
    print(minkowski_distance(vec1_test, vec2_test))

    print(euclidean_distance_standardized(vec1_test, vec2_test))
    print(mahalanobis_distance(vec1_test, vec2_test))

    print('###############################################')

    print(bray_curtis_distance(vec1_test, vec2_test))
    print(pearson_correlation_distance(vec1_test, vec2_test))
    print(jaccard_similarity_coefficient_distance(vec1_test, vec2_test))

    print('###############################################')

    # print(HamMings_Levenshtein(str1, str2)),需要等长
    # print(Wmd_distance(model, sent1_cut_list, sent2_cut_list)) # 需要gensim word2vec model

    print(hamming_distance(str1_test, str2_test))
    print(edit_levenshtein(str1_test, str2_test))
    print(ratio_levenshtein(str1_test, str2_test))
    print(jaro_levenshtein(str1_test, str2_test))
    print(set_ratio_fuzzywuzzy(str1_test, str2_test))
    print(sort_ratio_fuzzywuzzy(str1_test, str2_test))
    print(num_of_common_sub_str(str1_test, str2_test))
    print(normalization(vec1_test))  # 归一化（0-1）
    print(z_score(vec1_test))  # 标准化（0附近，正负）

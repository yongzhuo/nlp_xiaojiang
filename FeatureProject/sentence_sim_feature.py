# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/1 10:35
# @author   :Mo
# @function :calculate Similarity of text and vector, which are tf-idf and pinyin


from FeatureProject.distance_text_or_vec import euclidean_distance, cosine_distance, manhattan_distance, euclidean_distance, jaccard_similarity_coefficient_distance
from FeatureProject.distance_text_or_vec import chebyshev_distance, minkowski_distance, euclidean_distance_standardized
from FeatureProject.distance_text_or_vec import mahalanobis_distance, bray_curtis_distance, pearson_correlation_distance
from FeatureProject.distance_text_or_vec import wmd_distance, normalization, z_score
from FeatureProject.distance_text_or_vec import hamming_distance, edit_levenshtein, ratio_levenshtein, jaro_levenshtein, set_ratio_fuzzywuzzy, sort_ratio_fuzzywuzzy
from FeatureProject.distance_text_or_vec import clear_sentence, chinese2pinyin, num_of_common_sub_str
from conf.path_config import word2_vec_path, td_idf_path, td_idf_path_pinyin
from FeatureProject.distance_vec_TS_SS import TS_SS
from gensim import corpora, models, matutils
from conf.path_config import projectdir
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import pickle
import jieba
import time
import os


class SentenceSimFeature:
    def __init__(self):
        self.sen1 = None
        self.sen2 = None
        self.seg1 = None
        self.seg2 = None
        self.sen_vec1 = None
        self.sen_vec2 = None
        self.tfidf_vec1 = None
        self.tfidf_vec2 = None
        self.dictionary = None
        self.tfidf_model = None
        self.w2c_model = None

        self.tfidf_pinyin_model = None
        self.dictionary_pinyin = None
        self.sen1_pinyin = None
        self.sen2_pinyin = None
        self.seg1_pinyin = None
        self.seg2_pinyin = None
        self.tfidf_vec1_pinyin = None
        self.tfidf_vec2_pinyin = None

    def set_data(self, sen1, sen2):
        sen1 = clear_sentence(sen1)
        sen2 = clear_sentence(sen2)
        self.sen1 = str(sen1).strip()
        self.sen2 = str(sen2).strip()
        self.seg1 = list(jieba.cut(sen1))
        self.seg2 = list(jieba.cut(sen2))
        self.sen1_pinyin = chinese2pinyin(sen1)
        self.sen2_pinyin = chinese2pinyin(sen2)
        self.seg1_pinyin = (self.sen1_pinyin).split(' ')
        self.seg2_pinyin = (self.sen2_pinyin).split(' ')
        self.sen_vec1 = np.zeros(300)
        self.sen_vec2 = np.zeros(300)
        # self.tfidf_vec1 = np.array((self.tfidf_model.transform([' '.join(self.seg1)])).toarray().tolist()[0])
        # self.tfidf_vec2 = np.array((self.tfidf_model.transform([' '.join(self.seg2)])).toarray().tolist()[0])
        # self.tfidf_vec1_pinyin = np.array((self.tfidf_pinyin_model.transform([' '.join(self.seg1_pinyin)])).toarray().tolist()[0])
        # self.tfidf_vec2_pinyin = np.array((self.tfidf_pinyin_model.transform([' '.join(self.seg2_pinyin)])).toarray().tolist()[0])
        self.tfidf_vec1 = self.tfidf_model[self.dictionary.doc2bow(self.seg1)]
        self.tfidf_vec2 = self.tfidf_model[self.dictionary.doc2bow(self.seg2)]
        self.tfidf_vec1_pinyin = self.tfidf_pinyin_model[self.dictionary_pinyin.doc2bow(self.seg1_pinyin)]
        self.tfidf_vec2_pinyin = self.tfidf_pinyin_model[self.dictionary_pinyin.doc2bow(self.seg2_pinyin)]

    def same_word_count(self):
        count_left = 0
        for s in self.seg1:
            if s in self.seg2:
                count_left += 1

        count_right = 0
        for s in self.seg2:
            if s in self.seg1:
                count_right += 1

        return min(count_left, count_right)

    def same_char_count(self):
        seg1 = list(self.sen1)
        seg2 = list(self.sen2)

        count_left = 0
        for s in seg1:
            if s in seg2:
                count_left += 1

        count_right = 0
        for s in seg2:
            if s in seg1:
                count_right += 1

        return min(count_left, count_right)

    def sentence_length(self):
        len_sen1 = len(self.sen1)
        len_sen2 = len(self.sen2)
        len_abs_sub = abs(len_sen1 - len_sen2)
        len_rate = len_sen1 / len_sen2
        len_add_rate = len_sen1 * len_sen2 / (len_sen1 + len_sen2)

        return [len_abs_sub, len_rate, len_add_rate]

    def init_sentence_vector(self):
        # file_path = os.path.dirname(__file__)
        print('load w2v model begin')
        # model_path = os.path.join(file_path, word2_vec_path)
        self.w2c_model = KeyedVectors.load_word2vec_format(word2_vec_path, unicode_errors='ignore', limit=None)  # ,binary=True)
        print('load w2v model success')

    def encode_sentence_vector(self):
        for s in self.seg1:
            try:
                self.sen_vec1 += self.w2c_model[s]
            except:
                self.sen_vec1 += np.zeros(300)
                continue

        for s in self.seg2:
            try:
                self.sen_vec2 += self.w2c_model[s]
            except:
                self.sen_vec2 += np.zeros(300)
                continue

    def init_tfidf(self):
        file = open(td_idf_path, 'rb')
        tfidf_dictionary_model = pickle.load(file)
        self.dictionary = tfidf_dictionary_model[0]
        self.tfidf_model = tfidf_dictionary_model[1]

        file = open(td_idf_path_pinyin, 'rb')
        tfidf_dictionary_pinyin_model = pickle.load(file)
        self.dictionary_pinyin = tfidf_dictionary_pinyin_model[0]
        self.tfidf_pinyin_model = tfidf_dictionary_pinyin_model[1]
        print("init_tfidf ok!")

    def w2c_all_vec(self):
        w2c_Cosine = cosine_distance(self.sen_vec1, self.sen_vec2)
        w2c_TS_SS = TS_SS(self.sen_vec1, self.sen_vec2)
        w2c_Manhattan = manhattan_distance(self.sen_vec1, self.sen_vec2)
        w2c_Euclidean = euclidean_distance(self.sen_vec1, self.sen_vec2)
        w2c_Jaccard = jaccard_similarity_coefficient_distance(self.sen_vec1, self.sen_vec2)

        w2c_Chebyshev = chebyshev_distance(self.sen_vec1, self.sen_vec2)
        w2c_Minkowski = minkowski_distance(self.sen_vec1, self.sen_vec2)

        w2c_Euclidean_Standard = euclidean_distance_standardized(self.sen_vec1, self.sen_vec2)
        w2c_Mahalanobis = mahalanobis_distance(self.sen_vec1, self.sen_vec2)
        w2c_Bray = bray_curtis_distance(self.sen_vec1, self.sen_vec2)
        w2c_Pearson = pearson_correlation_distance(self.sen_vec1, self.sen_vec2)

        # w2c_Wmd = Wmd_Distance(self.w2c_model, self.sen_vec1, self.sen_vec2)
        return [w2c_Cosine, w2c_TS_SS, w2c_Manhattan, w2c_Euclidean, w2c_Jaccard, w2c_Chebyshev,
                w2c_Minkowski, w2c_Euclidean_Standard, w2c_Mahalanobis, w2c_Bray, w2c_Pearson]

    def tdidf_all_vec(self):

        return matutils.cossim(self.tfidf_vec1, self.tfidf_vec2)

    def edit_all_str(self):
        str_hamming = hamming_distance(self.sen1, self.sen2)
        str_edit = edit_levenshtein(self.sen1, self.sen2)
        str_ratio = ratio_levenshtein(self.sen1, self.sen2)
        str_jaro = jaro_levenshtein(self.sen1, self.sen2)
        str_set_ratio_fuzz = set_ratio_fuzzywuzzy(self.sen1, self.sen2)
        str_sort_ratio_fuzz = sort_ratio_fuzzywuzzy(self.sen1, self.sen2)
        str_commonsubstr = num_of_common_sub_str(self.sen1, self.sen2)
        str_list_Wmd = wmd_distance(self.w2c_model, self.seg1, self.seg2)

        return [str_hamming, str_edit, str_ratio, str_jaro,
                str_set_ratio_fuzz, str_sort_ratio_fuzz, str_commonsubstr, str_list_Wmd]

    def word_jaccard(self):
        a = list(set(self.seg1).intersection(set(self.seg2)))
        b = list(set(self.seg1).union(set(self.seg2)))
        return float(len(a) / len(b))

    def char_jaccard(self):
        a = list(set(list(self.sen1)).intersection(set(list(self.sen2))))
        b = list(set(list(self.sen1)).union(set(list(self.sen2))))

        return float(len(a) / len(b))

    def tdidf_all_vec_pinyin(self):

        return matutils.cossim(self.tfidf_vec1_pinyin, self.tfidf_vec2_pinyin)

    def edit_all_pinyin(self):
        pinyin_hamming = hamming_distance(self.sen1_pinyin, self.sen2_pinyin)
        pinyin_edit = edit_levenshtein(self.sen1_pinyin, self.sen2_pinyin)
        pinyin_ratio = ratio_levenshtein(self.sen1_pinyin, self.sen2_pinyin)
        pinyin_jaro = jaro_levenshtein(self.sen1_pinyin, self.sen2_pinyin)
        pinyin_set_ratio_fuzz = set_ratio_fuzzywuzzy(self.sen1_pinyin, self.sen2_pinyin)
        pinyin_sort_ratio_fuzz = sort_ratio_fuzzywuzzy(self.sen1_pinyin, self.sen2_pinyin)
        pinyin_commonsubstr = num_of_common_sub_str(self.sen1_pinyin, self.sen2_pinyin)
        pinyin_list_Wmd = wmd_distance(self.w2c_model, self.seg1_pinyin, self.seg2_pinyin)

        return [pinyin_hamming, pinyin_edit, pinyin_ratio, pinyin_jaro,
                pinyin_set_ratio_fuzz, pinyin_sort_ratio_fuzz, pinyin_commonsubstr, pinyin_list_Wmd]

    def word_jaccard_pinyin(self):
        a = list(set(self.seg1_pinyin).intersection(set(self.seg2_pinyin)))
        b = list(set(self.seg1_pinyin).union(set(self.seg2_pinyin)))
        return float(len(a) / len(b))

    def char_jaccard_pinyin(self):
        a = list(set(list(self.seg1_pinyin)).intersection(set(list(self.seg2_pinyin))))
        b = list(set(list(self.seg1_pinyin)).union(set(list(self.seg2_pinyin))))

        return float(len(a) / len(b))


def sentence_input_t():
    while True:
        s1 = input('s1: ')
        s2 = input('s2: ')

        start_time = time.time()
        ssf.set_data(s1, s2)
        ssf.encode_sentence_vector()

        time1 = time.time()
        print('set_data time：' + str(time1 - start_time))

        # 相同词、长度
        same_word_count = ssf.same_word_count()
        time2 = time.time()
        print('same_word_count time：' + str(time2 - time1))

        same_char_count = ssf.same_char_count()
        time3 = time.time()
        print('same_char_count time：' + str(time3 - time2))

        [len_abs_sub, len_rate, len_add_rate] = ssf.sentence_length()
        time4 = time.time()
        print('sentence_length time：' + str(time4 - time3))

        #  w2c_all_vec
        [w2c_Cosine, w2c_TS_SS, w2c_Manhattan, w2c_Euclidean,
         w2c_Jaccard, w2c_Chebyshev, w2c_Minkowski, w2c_Euclidean_Standard, w2c_Mahalanobis,
         w2c_Bray, w2c_Pearson] = ssf.w2c_all_vec()
        time5 = time.time()
        print('w2c_all_vec time：' + str(time5 - time4))

        #  tdidf_all_vec
        # [tdidf_Cosine, tdidf_TS_SS, tdidf_Manhattan, tdidf_Euclidean,
        #  tdidf_Jaccard, tdidf_Chebyshev,tdidf_Minkowski, tdidf_Euclidean_Standard, tdidf_Mahalanobis,
        #  tdidf_Bray, tdidf_Pearson] = ssf.tdidf_all_vec()
        tdidf_cossim = ssf.tdidf_all_vec()
        time6 = time.time()
        print('tdidf_all_vec time：' + str(time6 - time5))

        #  edit_all_str
        [str_hamming, str_edit, str_ratio, str_jaro,
         str_set_ratio_fuzz, str_sort_ratio_fuzz, str_commonsubstr, str_list_Wmd] = ssf.edit_all_str()
        time7 = time.time()
        print('edit_all_str time：' + str(time7 - time6))

        # jaccard系数
        word_jaccard = ssf.word_jaccard()
        char_jaccard = ssf.char_jaccard()
        time8 = time.time()
        print('jaccard系数 time：' + str(time8 - time7))

        #  tdidf_all_vec_pinyin
        # [tdidf_piyin_Cosine, tdidf_piyin_TS_SS, tdidf_piyin_Manhattan, tdidf_piyin_Euclidean, tdidf_piyin_Jaccard,
        #  tdidf_piyin_Chebyshev, tdidf_piyin_Minkowski, tdidf_piyin_Euclidean_Standard, tdidf_piyin_Mahalanobis,
        #  tdidf_piyin_Bray, tdidf_piyin_Pearson] = ssf.tdidf_all_vec_pinyin()
        tdidf_pinyin_cossim = ssf.tdidf_all_vec_pinyin()
        time9 = time.time()
        print('tdidf_all_vec_pinyin time：' + str(time9 - time8))

        #  edit_all_pinyin
        [pinyin_hamming, pinyin_edit, pinyin_ratio, pinyin_jaro,
         pinyin_set_ratio_fuzz, pinyin_sort_ratio_fuzz, pinyin_commonsubstr, pinyin_list_Wmd] = ssf.edit_all_pinyin()
        time10 = time.time()
        print('edit_all_pinyin time：' + str(time10 - time9))

        # jaccard系数
        word_jaccard_pinyin = ssf.word_jaccard_pinyin()
        char_jaccard_pinyin = ssf.char_jaccard_pinyin()
        time11 = time.time()
        print('jaccard系数pinyin  time：' + str(time11 - time10))
        sim_all_last = [same_word_count, same_char_count, len_abs_sub, len_rate, len_add_rate,
                        w2c_Cosine, w2c_TS_SS, w2c_Manhattan, w2c_Euclidean, w2c_Jaccard, w2c_Chebyshev, w2c_Minkowski,
                        w2c_Euclidean_Standard, w2c_Mahalanobis, w2c_Bray, w2c_Pearson,
                        tdidf_cossim, str_hamming, str_edit, str_ratio, str_jaro, str_set_ratio_fuzz,
                        str_sort_ratio_fuzz,
                        str_commonsubstr, str_list_Wmd,
                        word_jaccard, char_jaccard, tdidf_pinyin_cossim,
                        pinyin_hamming, pinyin_edit, pinyin_ratio, pinyin_jaro, pinyin_set_ratio_fuzz,
                        pinyin_sort_ratio_fuzz,
                        pinyin_commonsubstr, pinyin_list_Wmd,
                        word_jaccard_pinyin, char_jaccard_pinyin]
        print("sim: ")
        print(sim_all_last)


if __name__ == '__main__':
    ssf = SentenceSimFeature()
    ssf.init_sentence_vector()
    ssf.init_tfidf()
    s1 = "你知道Mo的能力上限吗"
    s2 = "你好呀，Mo水平很差"
    start_time = time.time()

    ssf.set_data(s1, s2)
    ssf.encode_sentence_vector()

    time1 = time.time()
    print('set_data time：' + str(time1 - start_time))

    # 相同词、长度
    same_word_count = ssf.same_word_count()
    time2 = time.time()
    print('same_word_count time：' + str(time2 - time1))

    same_char_count = ssf.same_char_count()
    time3 = time.time()
    print('same_char_count time：' + str(time3 - time2))

    [len_abs_sub, len_rate, len_add_rate] = ssf.sentence_length()
    time4 = time.time()
    print('sentence_length time：' + str(time4 - time3))

    #  w2c_all_vec
    [w2c_Cosine, w2c_TS_SS, w2c_Manhattan, w2c_Euclidean,
     w2c_Jaccard, w2c_Chebyshev, w2c_Minkowski, w2c_Euclidean_Standard, w2c_Mahalanobis,
     w2c_Bray, w2c_Pearson] = ssf.w2c_all_vec()
    time5 = time.time()
    print('w2c_all_vec time：' + str(time5 - time4))

    #  tdidf_all_vec
    tdidf_cossim = ssf.tdidf_all_vec()
    time6 = time.time()
    print('tdidf_all_vec time：' + str(time6 - time5))

    #  edit_all_str
    [str_hamming, str_edit, str_ratio, str_jaro,
     str_set_ratio_fuzz, str_sort_ratio_fuzz, str_commonsubstr, str_list_Wmd] = ssf.edit_all_str()
    time7 = time.time()
    print('edit_all_str time：' + str(time7 - time6))

    # jaccard系数
    word_jaccard = ssf.word_jaccard()
    char_jaccard = ssf.char_jaccard()
    time8 = time.time()
    print('jaccard系数 time：' + str(time8 - time7))

    # pinyin
    tdidf_pinyin_cossim = ssf.tdidf_all_vec_pinyin()
    time9 = time.time()
    print('tdidf_all_vec_pinyin time：' + str(time9 - time8))

    #  edit_all_pinyin
    [pinyin_hamming, pinyin_edit, pinyin_ratio, pinyin_jaro,
     pinyin_set_ratio_fuzz, pinyin_sort_ratio_fuzz, pinyin_commonsubstr, pinyin_list_Wmd] = ssf.edit_all_pinyin()
    time10 = time.time()
    print('edit_all_pinyin time：' + str(time10 - time9))

    # jaccard系数
    word_jaccard_pinyin = ssf.word_jaccard_pinyin()
    char_jaccard_pinyin = ssf.char_jaccard_pinyin()
    time11 = time.time()
    print('jaccard系数pinyin  time：' + str(time11 - time10))

    sim_all_last = [same_word_count, same_char_count, len_abs_sub, len_rate, len_add_rate,
                    w2c_Cosine, w2c_TS_SS, w2c_Manhattan, w2c_Euclidean, w2c_Jaccard, w2c_Chebyshev, w2c_Minkowski,
                    w2c_Euclidean_Standard, w2c_Mahalanobis, w2c_Bray, w2c_Pearson,
                    tdidf_cossim, str_hamming, str_edit, str_ratio, str_jaro, str_set_ratio_fuzz, str_sort_ratio_fuzz,
                    str_commonsubstr, str_list_Wmd,
                    word_jaccard, char_jaccard, tdidf_pinyin_cossim,
                    pinyin_hamming, pinyin_edit, pinyin_ratio, pinyin_jaro, pinyin_set_ratio_fuzz,
                    pinyin_sort_ratio_fuzz,
                    pinyin_commonsubstr, pinyin_list_Wmd,
                    word_jaccard_pinyin, char_jaccard_pinyin]
    print("小姜机器人计算sim: ")
    print(sim_all_last)

    sentence_input_t()

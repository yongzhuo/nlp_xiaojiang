# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/3 11:23
# @author   :Mo
# @function :path


import pathlib
import sys
import os


# base dir
projectdir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(projectdir)
print(projectdir)

# corpus
chicken_and_gossip_path = projectdir + '/Data/corpus/chicken_and_gossip.txt'

# word2vec
w2v_model_merge_short_path = projectdir + "/Data/chinese_vector/w2v_model_merge_short.vec"

# tf_idf
td_idf_cut_path = projectdir + '/Data/tf_idf/td_idf_cut.csv'
td_idf_cut_pinyin = projectdir + '/Data/tf_idf/td_idf_cut_pinyin.csv'
td_idf_path_pinyin = projectdir + '/Data/tf_idf/td_idf_cut_pinyin_dictionary_model.pkl'
td_idf_path = projectdir + '/Data/tf_idf/td_idf_cut_dictionary_model.pkl'

# word, 句向量
w2v_model_wiki_word_path = projectdir + '/Data/chinese_vector/w2v_model_wiki_word.vec'
matrix_ques_part_path = projectdir + '/Data/sentence_vec_encode_word/1.txt'

# char, 句向量
w2v_model_char_path = projectdir + '/Data/chinese_vector/w2v_model_wiki_char.vec'
matrix_ques_part_path_char = projectdir + '/Data/sentence_vec_encode_char/1.txt'

# word2vec select
word2_vec_path = w2v_model_wiki_word_path if os.path.exists(w2v_model_wiki_word_path) else w2v_model_merge_short_path

# stanford_corenlp_full_path
stanford_corenlp_full_path = "Y:/segment/stanford-corenlp-full-2018-10-05"

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

# stop_words_path
stop_words_path =  projectdir + '/Data/common_words/stopwords.txt'

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

# stanford_corenlp_full_path，需要自己下载配置stanford-corenlp-full-2018-10-05
stanford_corenlp_full_path = "Y:/segment/stanford-corenlp-full-2018-10-05"

# corpus webbank sim data char
train_data_web_char_dir = projectdir + '/AugmentText/augment_seq2seq/data_mid/char/'
train_data_web_ws_anti=projectdir + '/AugmentText/augment_seq2seq/data_mid/char/train_data_web_ws_anti.pkl'
train_data_web_xy_anti=projectdir + '/AugmentText/augment_seq2seq/data_mid/char/train_data_web_xy_anti.pkl'
model_ckpt_web_anti=projectdir + '/AugmentText/augment_seq2seq/model_seq2seq_tp/seq2seq_char_webank/model_ckpt_char_webank.ckp'
path_params=projectdir + '/conf/params.json'
path_webank_sim=projectdir + '/Data/corpus/sim_webank.csv'

# corpus webbank sim data word
train_data_web_word_dir = projectdir + '/AugmentText/augment_seq2seq/data_mid/word/'
train_data_web_emb_anti=projectdir + '/AugmentText/augment_seq2seq/data_mid/word/train_data_web_emb_anti.pkl'
train_data_web_xyw_anti=projectdir + '/AugmentText/augment_seq2seq/data_mid/word/train_data_web_ws_anti.pkl'
model_ckpt_web_anti_word=projectdir + '/AugmentText/augment_seq2seq/model_seq2seq_tp/seq2seq_word_webank/train_data_web_ws_anti.pkl'

# chatbot data char
chatbot_data_cg_char_dir = projectdir + '/ChatBot/chatbot_generate/seq2seq/data_mid/char/'
chatbot_data_cg_ws_anti=projectdir + '/ChatBot/chatbot_generate/seq2seq/data_mid/char/train_data_web_ws_anti.pkl'
chatbot_data_cg_xy_anti=projectdir + '/ChatBot/chatbot_generate/seq2seq/data_mid/char/train_data_web_xy_anti.pkl'
model_ckpt_cg_anti=projectdir + '/ChatBot/chatbot_generate/seq2seq/model_seq2seq_tp/seq2seq_char_cg/model_ckpt_char_cg.ckp'

# chatbot data word
chatbot_data_cg_word_dir = projectdir + '/ChatBot/chatbot_generate/seq2seq/data_mid/word/'
chatbot_data_cg_xyw_anti_word=projectdir + '/ChatBot/chatbot_generate/seq2seq/data_mid/word/train_data_cg_word_xyw.pkl'
chatbot_data_cg_emb_anti_word=projectdir + '/ChatBot/chatbot_generate/seq2seq/data_mid/word/train_data_cg_word_emb.pkl'
model_ckpt_cg_anti_word=projectdir + '/ChatBot/chatbot_generate/seq2seq/model_seq2seq_tp/seq2seq_word_cg/model_ckpt_word_cg.ckp'



# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/29 21:26
# @author   :Mo
# @function :同义词替换, word2vec版本

from utils.text_tools import txtRead, txtWrite, jieba_cut, load_word2vec_model
from conf.path_config import word2_vec_path
import logging as logger

word2vec_model = load_word2vec_model(word2_vec_path, limit_words=10000, binary_type=False, encoding_type='utf-8')


def select_all_syn_sentence(count=0, candidate_list_set=[], syn_sentences=[]):
    """
      递归函数，将形如 [['1'], ['1', '2'], ['1']] 的list转为 ['111','121']
    :param count: int, recursion times
    :param candidate_list_set: list, eg.[['你'], ['是', '是不是'], ['喜欢', '喜爱', '爱'], ['米饭']]
    :param syn_sentences: list, Storing intermediate variables of syn setnence, eg.['你是喜欢米饭', '你是不是喜欢米饭', '你是不是爱米饭']
    :return: list, result of syn setnence, eg.['你是喜欢米饭', '你是不是喜欢米饭', '你是不是爱米饭']
    """
    syn_sentences_new = []
    count = count - 1
    if count == -1:
        return syn_sentences
    for candidate_list_set_one in candidate_list_set[0]:
        for syn_sentences_one in syn_sentences:
            syn_sentences_new.append(syn_sentences_one + candidate_list_set_one)
    syn_sentences_new = select_all_syn_sentence(count=count, candidate_list_set=candidate_list_set[1:], syn_sentences=syn_sentences_new)
    return syn_sentences_new


def from_word2vec_get_synonym_words(sentence_cut, word2vec_model, count_words, top_words=20,
                                    topn_word_score=0.75):
    """
      jieba切词后替换为同义句, 从word2vec获取结果
    :param sentence_cut: list, demarcate words by jieba, eg.['你', '喜欢', '谁']
    :param words_list:   list, A pair of synonyms or same_words or antonyms, all ,eg.['良民\t顺民\n','汉人\t汉民\n']
    :param count_words:  int, statistics query times
    :param type:         boolean, Judging whether synonyms(True) or antonyms(False), eg.True or False
    :return: sentence_cut_dict(list, search words), count_words(int)
    """
    sentence_cut_dict = []
    count_word = 0
    for sentence_cut_one in sentence_cut:  # 切词后list中的一个词
        sentence_cut_dict_list = set()
        try:
            topn_words = word2vec_model.most_similar(sentence_cut_one, topn=top_words)
            for topn_word_num in topn_words:
                if topn_word_num[1] >= topn_word_score:
                    sentence_cut_dict_list.add(topn_word_num[0])
        except Exception as e:
            logger.info(str(e))

        if sentence_cut_dict_list:  # 如果有同义词或者反义词，就加上，如果没有，就加上自身
            sentence_cut_dict.append(list(sentence_cut_dict_list) + [sentence_cut_one])
            count_words.append(count_word)
        else:
            sentence_cut_dict.append([sentence_cut_one])
        count_word += 1

    return sentence_cut_dict, count_words

def word2vec_word_replace(sentence, top_words_put=20, topn_word_score_put=0.75):
    """
      只进行同义词替换，来生成同义句，同义词来源是word2vec
    :param sentence: str, input sentence of user, eg.'我喜欢谁你知道吗'
    :return: list, synonymous sentence generation

    """
    count_words = []
    sentence_cut = jieba_cut(sentence)
    len_sentence_cut = len(sentence_cut)
    count_word2vec_words = []
    # 根据切词结果list，按照word2vec模型获取同义词
    sentence_cut_word2vec_dict, count_word2vec_words = from_word2vec_get_synonym_words(sentence_cut,
                                                                                            word2vec_model,
                                                                                            count_words,
                                                                                            top_words=top_words_put,
                                                                                            topn_word_score=topn_word_score_put)

    # 根据获取到同义词，递归遍历生成同义句
    if len(sentence_cut_word2vec_dict) == 1:  # 没有同义词就返回原句子
        syn_sentence_cut_word2vec_list = sentence_cut
    syn_sentence_cut_word2vec_list = select_all_syn_sentence(count=len_sentence_cut - 1,
                                                                  candidate_list_set=sentence_cut_word2vec_dict[1:],
                                                                  syn_sentences=sentence_cut_word2vec_dict[0])
    syn_sentence_cut_word2vec_list.remove(sentence)

    return list(set(syn_sentence_cut_word2vec_list))


def get_syn_sentences(sentence_list, create_type='word2vec', top_words_put=20, topn_word_score_put=0.75):
        """
           批量生成同义句
        :param sentence_list: list, sentences of input, eg.['爱你', '你会什么']
        :param create_type: str, 'word2vec' or 'synonym'
        :param top_words_put: int, top n word of word2vec_model most_similar words
        :param topn_word_score_put: select topn_words of min most_similar score
        :return: 
        """
        syn_sentences = []
        # if create_type == 'word2vec':
        for sentence in sentence_list:
            syn_word2vec_one = word2vec_word_replace(sentence, top_words_put=20, topn_word_score_put=0.75)
            syn_sentences.append([sentence] + syn_word2vec_one)

        # todo 同义词典遍历生成
        return syn_sentences


def get_synonyms_from_word2vec(word2vec_model, word, topn=20, score_top=0.75):
    word_syn = []
    try:
        topn_words = word2vec_model.most_similar(word, topn=topn)
        for topn_word_num in topn_words:
            if topn_word_num[1] >= score_top:
                word_syn.append(topn_word_num[0])
    except Exception as e:
        logger.info(str(e))
    return word_syn

if __name__ == "__main__":
    sentence = '2005年美元换人民币的,'
    sentence_list = ['2006年美元换人民币']

    syns1 = word2vec_word_replace(sentence)
    print(syns1)
    print('#####################word2vec词典较小，需要自己在Data/chinese_vector新增，conf/payh_config.py需要修改word2_vec_path常量###############################')

    syn_sentences = get_syn_sentences(sentence_list)
    print(syn_sentences)
    print('###########################################################')
    while True:
        print('input: ')
        sen = input()
        syns1 = word2vec_word_replace(sen)
        print('###########################word2vec_words ###########################################################')
        print(syns1)

# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @Time     :2019/3/25 14:11
# @author   :Mo
# @function :generate disorder sentence by marko

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from conf.path_config import chicken_and_gossip_path
from conf.path_config import projectdir
from utils.text_tools import txtRead
from utils.text_tools import txtWrite
from jieba import analyse
import random
import jieba


# 引入TF-IDF关键词抽取接口
tfidf = analyse.extract_tags
# 引入TextRank关键词抽取接口
textrank = analyse.textrank


def create_model(model_markov, datalist):
    """
      create model of sentence sequence
    :param model_marko: dict
    :param datalist: list of set
    :return: set
    """
    for line in datalist:
        line = list(jieba.cut(line.lower().strip(), cut_all=False))
        for i, word in enumerate(line):
            if i == len(line) - 1:
                model_markov['FINISH'] = model_markov.get('FINISH', []) + [word]
            else:
                if i == 0:
                    model_markov['BEGIN'] = model_markov.get('BEGIN', []) + [word]
                model_markov[word] = model_markov.get(word, []) + [line[i + 1]]

    for key in model_markov.keys():
        model_markov[key] = list(set(model_markov[key]))

    return model_markov


def generate_random_1(model_markov, gen_words):
    """
       根据马尔科夫链生成同义句，本质就是根据一个词走到另外一个词去
    :param generated: list, empty
    :param model_marko: dict, marko of dict
    :return: str
    """
    while True:
        if not gen_words:
            words = model_markov['BEGIN']
        elif gen_words[-1] in model_markov['FINISH']:
            break
        else:
            try:
                words = model_markov[gen_words[-1]]
            except Exception as e:
                return "".join(gen_words) + "\n"
        # 随机选择一个词语
        gen_words.append(random.choice(words))

    return "".join(gen_words) + "\n"


def generate_random_select(generated, model_marko, twice=100000, len_min=5):
    """
      默认遍历1000次生成句子
    :param generated: list, one key word, rg.["建行"]
    :param model_marko: dict, transition matrix
    :param twice: int, twice
    :param len_min: int, min length of gen sentence 
    :return: list, syn_generates
    """
    syn_generates = set()
    for num in range(twice):
        syn_generate = generate_random_1(model_marko, generated)
        generated = []
        if len(syn_generate) > len_min:
            syn_generates.add(syn_generate)
    return list(syn_generates)


def get_keyword_from_tf(sentences, p):
    """
      获取某个类型下语料的热词
    :param sentences: list, cuted sentences, filter by " "
    :param p: float, rate, 0 < p < 1
    :return: list, words
    """
    sentence_cut_list = [" ".join(list(jieba.cut(text.strip(), cut_all=False, HMM=True))) for text in sentences]
    # token_pattern指定统计词频的模式, 不指定, 默认如英文, 不统计单字
    vectorizer = CountVectorizer(token_pattern='\\b\\w+\\b')
    # norm=None对词频结果不归一化
    # use_idf=False, 因为使用的是计算tfidf的函数, 所以要忽略idf的计算
    transformer = TfidfTransformer(norm=None, use_idf=False)
    vectorizer.fit_transform(sentence_cut_list)
    # tf = transformer.fit_transform(vectorizer.fit_transform(sentence_cut_list))
    word = vectorizer.get_feature_names()
    # weight = tf.toarray()
    return word[-int(len(word) * p):]


def get_begin_word(sentences, p):
    """
      获取jieba切词后
    :param sentences:list, sentences of input 
    :param p: float, 
    :return: list, key_words
    """
    sentence_cut_begin_list = [list(jieba.cut(text.strip(), cut_all=False, HMM=True))[0] for text in sentences]
    len_begin_p = int(len(sentence_cut_begin_list) * p)
    return sentence_cut_begin_list[-len_begin_p:]


def get_keyword_from_jieba_tfidf(sentences, p):
    """
      基于TF-IDF算法进行关键词抽取
    :param sentence: str, sentence of input
    :return: list, return keyword
    """
    sentence_cut_list = [" ".join(list(jieba.cut(text.strip(), cut_all=False, HMM=True))) for text in sentences]
    sentence_cut_list_str = str(sentence_cut_list)
    key_word = tfidf(sentence_cut_list_str)
    return key_word


def get_keyword_from_jieba_textrank(sentences, p):
    """
      基于textrank算法进行关键词抽取
    :param sentence: str, sentence of input
    :return: list, return keyword
    """
    key_words = []
    for sentences_one in sentences:
        key_word = textrank(sentences_one)
        key_words = key_words + key_word
    # token_pattern指定统计词频的模式, 不指定, 默认如英文, 不统计单字
    vectorizer = CountVectorizer(token_pattern='\\b\\w+\\b')
    vectorizer.fit_transform(key_words)
    word = vectorizer.get_feature_names()
    return word[-int(len(word) * p):]


def generate_syns_from_list(sentence_list, begin_word="tfidf", p=0.1):
    """
      读取txt文件原语句，获取没有的生成句子
    :param txt_path: str, path of corpus
    :param begin_word: str, "tf", "tfidf", "textrank"
    :param p: float, rate, 0 < p < 1 
    :return: list, generated sentence
    """
    # 获取热门关键词
    if begin_word == "tf":
        generated_hot = get_keyword_from_tf(sentence_list, p)
    elif begin_word == "textrank":
        generated_hot = get_keyword_from_jieba_textrank(sentence_list, p)
    elif begin_word == "begin_word":
        generated_hot = get_begin_word(sentence_list, p)
    else:
        generated_hot = get_keyword_from_jieba_tfidf(sentence_list, p)

    # 创建传递模型
    model_txt = {}
    model_txt = create_model(model_txt, sentence_list)
    # 以关键词开头，构建同义句
    gen_all_syn = []
    for generated_hot_one in generated_hot:
        generated_hot_one_1 = [generated_hot_one]
        generated_str = generate_random_select(generated_hot_one_1, model_txt, twice=1000, len_min=5)
        if generated_str:
            gen_all_syn = gen_all_syn + generated_str
    # 提取原句中没有的部分
    gen_all_syn = list(set(gen_all_syn))
    # 生成句子与原句的交集
    syn_intersection = list(set(sentence_list).intersection(set(gen_all_syn)))
    # 生成句子减去交集
    gen_syns = list(set(gen_all_syn).difference(set(syn_intersection)))
    return gen_syns


if __name__ == "__main__":
    # 读取一个文件，再生成句子
    txt_path = chicken_and_gossip_path
    sentence_list = txtRead(txt_path)
    sentence_list = sentence_list[0:100]
    enhance_texts = generate_syns_from_list(sentence_list, begin_word="tfidf", p=0.1)
    for enhance_texts_one in enhance_texts:
        try:
            print(enhance_texts_one)
        except Exception as e:
            print(str(e))
# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/1 10:35
# @author   :Mo
# @function :cut sentences


from conf.path_config import chicken_and_gossip_path, td_idf_cut_path, td_idf_cut_pinyin
from utils.text_tools import txtWrite, txtRead, get_syboml, strQ2B
from conf.path_config import projectdir
from gensim import corpora, models
import xpinyin
import pickle
import jieba


def cut_td_idf(sources_path, target_path):
    """
    结巴切词，汉语
    :param path: 
    :return: 
    """
    print("cut_td_idf start! ")
    corpus = txtRead(sources_path)
    governments = []
    for corpus_one in corpus:
        corpus_one_clear = corpus_one.replace(' ', '').strip()
        ques_q2b = strQ2B(corpus_one_clear.strip())
        ques_q2b_syboml = get_syboml(ques_q2b)
        governments.append(ques_q2b_syboml.strip())

    government_ques = list(map(lambda x: ' '.join(jieba.lcut(x)), governments))

    topic_ques_all = []
    for topic_ques_one in government_ques:
        top_ques_aqlq = topic_ques_one.replace('   ', ' ').replace('  ', ' ').strip() + '\n'
        topic_ques_all.append(top_ques_aqlq)

    txtWrite(topic_ques_all, target_path)
    print("cut_td_idf ok! " + sources_path)


def cut_td_idf_pinyin(sources_path, target_path): #获取拼音
    """
       汉语转拼音
    :param path: 
    :return: 
    """
    pin = xpinyin.Pinyin()
    corpus = txtRead(sources_path)
    topic_ques_all = []
    corpus_count = 0
    for corpus_one in corpus:
        corpus_count += 1
        # time1 = time.time()
        corpus_one_clear = corpus_one.replace(' ', '').strip()
        ques_q2b = strQ2B(corpus_one_clear.strip())
        ques_q2b_syboml = get_syboml(ques_q2b)
        ques_q2b_syboml_pinying = pin.get_pinyin(ques_q2b_syboml.replace('   ', '').replace('  ', '').strip(), ' ')
        topic_ques_all.append(ques_q2b_syboml_pinying + '\n')
        # time2 = time.time()
        # print(str(corpus_count) + 'time:' + str(time2 - time1))
    txtWrite(topic_ques_all, target_path)
    print("cut_td_idf_pinyin ok! " + sources_path)


def init_tfidf_chinese_or_pinyin(sources_path):
    """
      构建td_idf
    :param path: 
    :return: 
    """
    questions = txtRead(sources_path)
    corpora_documents = []
    for item_text in questions:
        item_seg = list(jieba.cut(str(item_text).strip()))
        corpora_documents.append(item_seg)

    dictionary = corpora.Dictionary(corpora_documents)
    corpus = [dictionary.doc2bow(text) for text in corpora_documents]
    tfidf_model = models.TfidfModel(corpus)
    print("init_tfidf_chinese_or_pinyin ok! " + sources_path)
    file = open(sources_path.replace(".csv", "_dictionary_model.pkl"), 'wb')
    pickle.dump([dictionary, tfidf_model], file)


if __name__ == '__main__':
    # path_text = projectdir + '/Data/chicken_gossip.txt'
    # sentences = txtRead(path_text)
    # sentences_q = []
    # for sentences_one in sentences:
    #     sentences_one_replace = sentences_one.replace(" ", "").replace("\t", "")
    #     sentences_one_replace_split = sentences_one_replace.split("|")
    #     sentence_new = sentences_one_replace_split[0] + "\t" + "".join(sentences_one_replace_split[1:])
    #     sentences_q.append(sentence_new)
    # sentences = txtWrite(sentences_q, projectdir + '/Data/chicken_and_gossip.txt')


    cut_td_idf(chicken_and_gossip_path, td_idf_cut_path)
    cut_td_idf_pinyin(chicken_and_gossip_path, td_idf_cut_pinyin)
    init_tfidf_chinese_or_pinyin(td_idf_cut_path)
    init_tfidf_chinese_or_pinyin(td_idf_cut_pinyin)
    print("corpus ok!")


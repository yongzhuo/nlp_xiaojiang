# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/5/12 13:16
# @author   :Mo
# @function :chatbot based search, encode sentence_vec by bert

def chatbot_sentence_vec_by_bert_own():
    """bert encode is writted by my own"""
    from FeatureProject.bert.extract_keras_bert_feature import KerasBertVector
    from conf.path_config import chicken_and_gossip_path
    from utils.text_tools import txtRead
    import numpy as np

    # 读取数据和一些参数，这里只取了100个标准问题
    topk = 5
    matrix_ques_save_path = "doc_vecs_chicken_and_gossip"
    questions = txtRead(chicken_and_gossip_path, encodeType='utf-8')
    ques = [ques.split('\t')[0] for ques in questions][0:100]

    # 生成标准问题的bert句向量
    bert_vector = KerasBertVector()
    ques_basic_vecs = bert_vector.bert_encode(ques)

    # 线上你可以生成，直接调用，然后直接load就好
    np.savetxt(matrix_ques_save_path, ques_basic_vecs)
    # matrix_ques = np.loadtxt(matrix_ques_save_path)

    query_bert_vec = bert_vector.bert_encode(["小姜机器人是什么"])[0]
    query_bert_vec = np.array(query_bert_vec)
    print(query_bert_vec)
    # 矩阵点乘，很快的，你也可以用annoy等工具，计算就更加快了
    qq_score = np.sum(query_bert_vec * ques_basic_vecs, axis=1) / np.linalg.norm(ques_basic_vecs, axis=1)
    topk_idx = np.argsort(qq_score)[::-1][:topk]
    for idx in topk_idx:
        print('小姜机器人回答检索： %s\t%s' % (qq_score[idx], questions[idx]))


    while True:
        print("你的问题:")
        query = input()
        query_bert_vec = bert_vector.bert_encode([query])[0]
        query_bert_vec = np.array(query_bert_vec)
        # 矩阵点乘，很快的，你也可以用annoy等工具，计算就更加快了
        qq_score = np.sum(query_bert_vec * ques_basic_vecs, axis=1) / np.linalg.norm(ques_basic_vecs, axis=1)
        topk_idx = np.argsort(qq_score)[::-1][:topk]
        for idx in topk_idx:
            print('小姜机器人回答检索： %s\t%s' % (qq_score[idx], questions[idx]))


def chatbot_sentence_vec_by_bert_bertasserver():
    """bert encode is used bert as server"""
    from conf.path_config import chicken_and_gossip_path
    from bert_serving.client import BertClient
    from utils.text_tools import txtRead
    import numpy as np

    topk = 5
    matrix_ques_save_path = "doc_vecs_chicken_and_gossip"
    questions = txtRead(chicken_and_gossip_path, encodeType='utf-8')
    ques = [ques.split('\t')[0] for ques in questions][0:100]

    bc = BertClient(ip = 'localhost')
    doc_vecs = bc.encode(ques)
    np.savetxt(matrix_ques_save_path, doc_vecs)
    # matrix_ques = np.loadtxt(matrix_ques_save_path)

    while True:
        query = input('你问: ')
        query_vec = bc.encode([query])[0]
        query_bert_vec = np.array(query_bert_vec)
        # compute normalized dot product as score
        score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]
        for idx in topk_idx:
            print('小姜机器人回答： %s\t%s' % (score[idx], questions[idx]))


if __name__=="__main__":
    chatbot_sentence_vec_by_bert_own()
    # chatbot_sentence_vec_by_bert_bertasserver()


# result
# 小姜机器人是什么
# Tokens: ['[CLS]', '小', '姜', '机', '器', '人', '是', '什', '么', '[SEP]']
# (1, 32, 768)
# [CLS] [768] [1.0393640995025635, -0.31394684314727783, -0.08567211031913757, -0.12281288206577301,

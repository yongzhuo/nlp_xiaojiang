# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/5/7 20:27
# @author   :Mo
# @function :test sentence of bert encode and cosin sim of two question


def calculate_count():
    """
      统计一下1000条测试数据的平均耗时
    :return: 
    """
    from FeatureProject.bert.extract_keras_bert_feature import KerasBertVector
    import time

    bert_vector = KerasBertVector()
    print("bert start ok!")
    time_start = time.time()
    for i in range(1000):
        vector = bert_vector.bert_encode(["jy，你知道吗，我一直都很喜欢你呀，在一起在一起在一起，哈哈哈哈"])

    time_end = time.time()
    time_avg = (time_end-time_start)/1000
    print(vector)
    print(time_avg)
    # 0.12605296468734742  win10 gpu avg
    # 0.01629048466682434  linux cpu avg


def sim_two_question():
    """测试一下两个问题的相似句子"""
    from FeatureProject.bert.extract_keras_bert_feature import KerasBertVector
    from sklearn import preprocessing
    from math import pi
    import numpy as np
    import time
    import math

    def cosine_distance(v1, v2): # 余弦距离
        if v1.all() and v2.all():
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        else:
            return 0

    def scale_zoom(rate): # sig 缩放
        zoom = (1 + np.exp(-float(rate))) / 2
        return zoom

    def scale_triangle(rate): # sin 缩放
        triangle = math.sin(rate/1*pi/2 - pi/2)
        return triangle

    bert_vector = KerasBertVector()
    print("bert start ok!")
    while True:
        print("input ques-1: ")
        ques_1 = input()
        print("input ques_2: ")
        ques_2 = input()
        vector_1 = bert_vector.bert_encode([ques_1])
        vector_2 = bert_vector.bert_encode([ques_2])
        sim = cosine_distance(vector_1[0], vector_2[0])
        # sim_list = [sim, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # sim = preprocessing.scale(sim_list)[0]
        # sim = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(sim_list)[0]
        # sim_1 = preprocessing.normalize(sim_list, norm='l1')[0]
        # sim_2 = preprocessing.normalize(sim_list, norm='l2')[0]
        # sim = scale_zoom(sim)
        # sim = scale_triangle(sim)
        # print(sim_1)
        # print(sim_2)
        print(sim)


if __name__=="__main__":
    calculate_count()
    sim_two_question()

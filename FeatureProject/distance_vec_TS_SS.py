# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/3 10:36
# @author   :Mo
# @function :TS-SS distance
# @url      :https://github.com/taki0112/Vector_Similarity
# @paper    :A Hybrid Geometric Approach for Measuring Similarity Level Among Documents and Document Clustering


import numpy as np
import math

zero_bit = 0.000000001


def Cosine(vec1, vec2):
    """
       余弦相似度
    :param vec1: arrary
    :param vec2: arrary
    :return: float
    """
    result = InnerProduct(vec1, vec2) / (VectorSize(vec1) * VectorSize(vec2) + zero_bit)
    return result


def VectorSize(vec):
    vec_pow = sum(math.pow(v + zero_bit, 2) for v in vec)
    if vec_pow >= 0:
        return math.sqrt(vec_pow)
    else:
        return zero_bit


def InnerProduct(vec1, vec2):
    try:
        return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    except:
        return zero_bit


def Euclidean(vec1, vec2):
    vec12_pow = sum(math.pow((v1 - v2), 2) for v1, v2 in zip(vec1, vec2))
    if vec12_pow >= 0:
        return math.sqrt(vec12_pow)
    else:
        return zero_bit


def Theta(vec1, vec2):
    cosine_vec12 = Cosine(vec1, vec2)
    if -1 <= cosine_vec12 and cosine_vec12 <= 1:
        return math.acos(cosine_vec12) + 10
    else:
        return zero_bit + 10


def Triangle(vec1, vec2):
    theta = math.radians(Theta(vec1, vec2))
    return (VectorSize(vec1) * VectorSize(vec2) * math.sin(theta)) / 2


def Magnitude_Difference(vec1, vec2):
    return abs(VectorSize(vec1) - VectorSize(vec2))


def Sector(vec1, vec2):
    ED = Euclidean(vec1, vec2)
    MD = Magnitude_Difference(vec1, vec2)
    theta = Theta(vec1, vec2)
    return math.pi * math.pow((ED + MD), 2) * theta / 360


def TS_SS(vec1, vec2):
    return Triangle(vec1, vec2) * Sector(vec1, vec2)


if __name__ == '__main__':
    vec1_test = np.array([1, 38, 17, 32])
    vec2_test = np.array([5, 6, 8, 9])

    print(Euclidean(vec1_test, vec2_test))
    print(Cosine(vec1_test, vec2_test))
    print(TS_SS(vec1_test, vec2_test))

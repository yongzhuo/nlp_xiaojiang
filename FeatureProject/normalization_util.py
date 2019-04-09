# -*- coding: UTF-8 -*-
#!/usr/bin/python
# @Time     :2019/3/12 14:18
# @author   :Mo
# @site     :https://blog.csdn.net/rensihui

from sklearn import preprocessing
import numpy as np

def autoL1L2(data, norms = 'l1'):
    '''L1或者L2正则化'''
    return preprocessing.normalize(data, norm = norms)

def autoScale(data):
    '''标准化, (X-mean)/std.得到的结果是，对于每个属性/每列来说所有数据都聚集在0附近，方差为1。'''
    return preprocessing.scale(data)

def autoMinMaxScaler(data):
    '''将属性缩放到一个指定范围'''
    return preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(data)

def autoLinNorm(data):  # 传入一个矩阵
    ''' 0-1归一化
        :param data: []矩阵
        :return:     []
    '''
    mins = data.min(0)  # 返回data矩阵中每一列中最小的元素，返回一个列表
    maxs = data.max(0)  # 返回data矩阵中每一列中最大的元素，返回一个列表
    ranges = maxs - mins  # 最大值列表 - 最小值列表 = 差值列表
    normData = np.zeros(np.shape(data))  # 生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row = data.shape[0]  # 返回 data矩阵的行数
    normData = data - np.tile(mins, (row, 1))  # data矩阵每一列数据都减去每一列的最小值
    normData = normData / np.tile(ranges, (row, 1))  # data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
    return normData



def autoAvgNorm(data):  # 传入一个矩阵
    ''' 均值归一化
        :param data: []矩阵
        :return:     []
    '''
    avg = np.average(data, axis=1)  # 返回data矩阵中每一列中最小的元素，返回一个列表
    sigma = np.std(data, axis=1)  # 返回data矩阵中每一列中最大的元素，返回一个列表
    normData = np.zeros(np.shape(data))  # 生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row = data.shape[0]  # 返回 data矩阵的行数
    normData = data - np.tile(avg, (row, 1))  # data矩阵每一列数据都减去每一列的最小值
    normData = normData / np.tile(sigma, (row, 1))  # data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
    return normData



###Sigmoid函数；Sigmoid函数是一个具有S形曲线的函数，是良好的阈值函数，在(0, 0.5)处中心对称，在(0, 0.5)附近有比较大的斜率，
# 而当数据趋向于正无穷和负无穷的时候，映射出来的值就会无限趋向于1和0，是个人非常喜欢的“归一化方法”，之所以打引号是因为我觉得Sigmoid函数在
# 阈值分割上也有很不错的表现，根据公式的改变，就可以改变分割阈值，这里作为归一化方法，我们只考虑(0, 0.5)作为分割阈值的点的情况：
def sigmoid(data,useStatus):
    '''  sig归一化
        :param data: []矩阵
        :return:     []
    '''
    if useStatus:
        row=data.shape[0]
        column=data.shape[1]
        normData = np.zeros(np.shape(data))
        for i in range(row):
            for j in range(column):
                normData[i][j]=1.0 / (1 + np.exp(-float(data[i][j])));
        return normData
    else:
        return float(data);

if __name__ == '__main__':
    arr = np.array([[8, 7, 8], [4, 3, 1], [6, 9, 8]])

    print("l1正则化")
    print(autoL1L2(arr, norms='l1'))

    print("l2正则化")
    print(autoL1L2(arr, norms='l2'))

    print("0-1标准化处理")
    print(autoScale(arr))

    print("0-1缩放处理")
    print(autoMinMaxScaler(arr))


    print("0-1归一化处理")
    print(autoLinNorm(arr))


    print("均值归一化处理")
    print(autoAvgNorm(arr))

    print("sig归一化处理")
    print(sigmoid(arr,True))

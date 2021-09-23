# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/9/22 21:29
# @author  : Mo
# @function: 获取下三角矩阵


import keras.backend as K
import numpy as np

# np.random.rand(2,3)

input_x = np.array([[1,2,3], [4,5,6], [7,8,9]])
s = K.cast(input_x, dtype="float32")
idxs = K.cumsum(s, axis=1)  # 一行一行累加, 可用于构建上三角矩阵、下三角矩阵
print(K.eval(idxs))
mask = idxs[:, None, :] <= idxs[:, :, None]
print(K.eval(mask))
mask = K.cast(mask, K.floatx())
print(K.eval(mask))
ee = 0

print(75.27 / 20)


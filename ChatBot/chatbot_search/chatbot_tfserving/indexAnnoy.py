# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/4/18 21:04
# @author  : Mo
# @function: annoy search


from annoy import AnnoyIndex
import numpy as np
import os


class AnnoySearch:
    def __init__(self, dim=768, n_cluster=100):
        # metric可选“angular”（余弦距离）、“euclidean”（欧几里得距离）、 “ manhattan”（曼哈顿距离）或“hamming”（海明距离）
        self.annoy_index = AnnoyIndex(dim, metric="angular")
        self.n_cluster = n_cluster
        self.dim = dim

    def k_neighbors(self, vectors, k=18):
        """ 搜索 """
        annoy_tops = []
        for v in vectors:
            idx, dist = self.annoy_index.get_nns_by_vector(v, k, search_k=32*k, include_distances=True)
            annoy_tops.append([dist, idx])
        return annoy_tops

    def fit(self, vectors):
        """ annoy构建 """
        for i, v in enumerate(vectors):
            self.annoy_index.add_item(i, v)
        self.annoy_index.build(self.n_cluster)

    def save(self, path):
        """ 存储 """
        self.annoy_index.save(path)

    def load(self, path):
        """ 加载 """
        self.annoy_index.load(path)


if __name__ == '__main__':
    ### 索引
    import random
    path = "model.ann"
    dim = 768
    vectors = [[random.gauss(0, 1) for z in range(768)] for i in range(10)]
    an_model = AnnoySearch(dim, n_cluster=32)  # Length of item vector that will be indexed
    an_model.fit(vectors)
    an_model.save(path)
    tops = an_model.k_neighbors([vectors[0]], 18)
    print(tops)

    del an_model

    ### 下载, 搜索
    an_model = AnnoySearch(dim, n_cluster=32)
    an_model.load(path)
    tops = an_model.k_neighbors([vectors[0]], 6)
    print(tops)



    """ 
    # example
    from annoy import AnnoyIndex
    import random

    dim = 768
    vectors = [[random.gauss(0, 1) for z in range(768)] for i in range(10)]
    ann_model = AnnoyIndex(dim, 'angular')  # Length of item vector that will be indexed
    for i,v in enumerate(vectors):
        ann_model.add_item(i, v)
    ann_model.build(10)  # 10 trees
    ann_model.save("tet.ann")
    del ann_model

    u = AnnoyIndex(dim, "angular")
    u.load('tet.ann')  # super fast, will just mmap the file
    v = vectors[1]
    idx, dist = u.get_nns_by_vector(v, 10, search_k=50 * 10, include_distances=True)
    print([idx, dist])
    """



###  备注说明: annoy索引 无法 增删会改查




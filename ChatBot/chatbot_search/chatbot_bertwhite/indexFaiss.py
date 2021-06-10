# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/5/9 16:02
# @author  : Mo
# @function: search of faiss


from faiss import normalize_L2
import numpy as np
import faiss
import os


class FaissSearch:
    def __init__(self, dim=768, n_cluster=100):
        self.n_cluster = n_cluster  # 聚类中心
        self.dim = dim
        quantizer = faiss.IndexFlatIP(self.dim)
        #  METRIC_INNER_PRODUCT:余弦; L2: faiss.METRIC_L2
        self.faiss_index = faiss.IndexIVFFlat(quantizer, self.dim, self.n_cluster, faiss.METRIC_INNER_PRODUCT)
        # self.faiss_index = faiss.IndexFlatIP(self.dim)  # 索引速度更快 但是不可增量

    def k_neighbors(self, vectors, k=6):
        """ 搜索 """
        normalize_L2(vectors)
        dist, index = self.faiss_index.search(vectors, k)  # sanity check
        return dist.tolist(), index.tolist()

    def fit(self, vectors):
        """ annoy构建 """
        normalize_L2(vectors)
        self.faiss_index.train(vectors)
        # self.faiss_index.add(vectors)
        self.faiss_index.add_with_ids(vectors, np.arange(0, len(vectors)))

    def remove(self, ids):
        self.faiss_index.remove_ids(np.array(ids))

    def save(self, path):
        """ 存储 """
        faiss.write_index(self.faiss_index, path)

    def load(self, path):
        """ 加载 """
        self.faiss_index = faiss.read_index(path)


if __name__ == '__main__':

    import random

    path = "model.fai"
    dim = 768
    vectors = np.array([[random.gauss(0, 1) for z in range(768)] for i in range(32)], dtype=np.float32)
    fai_model = FaissSearch(dim, n_cluster=32)    # Length of item vector that will be indexed
    fai_model.fit(vectors)
    fai_model.save(path)
    tops = fai_model.k_neighbors(vectors[:32], 32)
    print(tops)
    ids = np.arange(10, 32)
    fai_model.remove(ids)
    tops = fai_model.k_neighbors(vectors[:32], 32)
    print(tops)
    print(len(tops))

    del fai_model

    fai_model = FaissSearch(dim, n_cluster=32)
    fai_model.load(path)
    tops = fai_model.k_neighbors(vectors[:32], 32)
    print(tops)



    """
    import numpy as np
    d = 64                           # dimension
    nb = 100000                      # database size
    nq = 10000                       # nb of queries
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    
    import faiss                   # make faiss available
    # # 量化器索引
    # nlist = 1000  # 聚类中心的个数
    # k = 50  # 邻居个数
    # quantizer = faiss.IndexFlatIP(d)  # the other index，需要以其他index作为基础
    # index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)  # METRIC_INNER_PRODUCT:余弦; L2: faiss.METRIC_L2
    
    ntree = 132  # 聚类中心的个数
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, ntree, faiss.METRIC_INNER_PRODUCT)
    # index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    index.add(xb)                  # add vectors to the index
    print(index.ntotal)
    
    k = 4                          # we want to see 4 nearest neighbors
    D, I = index.search(xb[:5], k) # sanity check
    print(I)
    print(D)
    D, I = index.search(xq, k)     # actual search
    print(I[:5])                   # neighbors of the 5 first queries
    print(I[-5:])                  # neighbors of the 5 last queries
    """


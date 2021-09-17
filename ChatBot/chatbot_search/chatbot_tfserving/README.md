# 新增一个余弦相似度Cosine层, 用于BERT句向量编码部署tf-serving
## 业务需求
 - BERT向量召回问答对, FAQ标准问答对数据量不大
 - 不能把BERT编码部署于网络服务, 如http请求的形式, 因为网络传输耗时, 此外传输的数据量还很大768(维度)*32(float) 
 - 几乎所有的模型服务只能用cpu, 硬盘、内存都还可以
 - 响应要求高, 小时延不能太高

## 代码逻辑
 - 首先将FAQ标准问答对生成句向量, bert-sentence-encode;
 - 将句向量当成一个 常量 插入网络, 网络架构新增 余弦相似度层(CosineLayer) 模块, 保存成tf-serving形式;
 - 选择小模型tinyBERT， ROBERTA-4-layer, ROBERTA-6-layer这些模型

## 解释说明
 - 代码说明:
  - TFServing_main.py         主代码, 调用
  - TFServing_postprocess.py  tf-serving 后处理函数
  - TFServing_preprocess.py   tf-serving 预处理函数
  - TFServing_save.py         tf-serving 主调用函数
 - 主调用
  - 1. bertWhiteConf.py  超参数配置, 地址、bert-white、索引工具等的超参数
  - 2. bertWhiteTools.py 小工具, 主要是一些文档读写功能函数
  - 3. bertWhiteTrain.py 主模块, 类似bert预训练模型编码
  - 4. indexAnnoy.py     annoy索引
  - 5. indexFaiss.py     faiss索引
  - 6. mmr.py            最大边界相关法, 保证返回多样性
  
## 模型文件
 - bert_white文件         bertWhiteTrain.py生成的模块
 - chatbot_tfserving文件  包含相似度计算的tf-serving文件
  
## 调用示例
 - 配置问答语料文件(chicken_and_gossip.txt) 和 超参数(bertWhiteConf.py中的BERT_DIR)
 - 生成FAQ句向量:               python3 bertWhiteTrain.py
 - 存储成pd文件(tf-serving使用): python3 TFServing_save.py
 - 部署docker服务(tf-serving):  例如 docker run -t --rm -p 8532:8501 -v "/TF-SERVING/chatbot_tf:/models/chatbot_tf" -e MODEL_NAME=chatbot_tf tensorflow/serving:latest
 - 调用tf-serving服务:          python3 TFServing_tet_http.py

## 关键代码
```python3
import keras.backend as K
import tensorflow as tf
import keras

import numpy as np


class CosineLayer(keras.layers.Layer):
    def __init__(self, docs_encode, **kwargs):
        """
        余弦相似度层, 不适合大规模语料, 比如100w以上的问答对
        :param docs_encode: np.array, bert-white vector of senence 
        :param kwargs: 
        """
        self.docs_encode = docs_encode
        super(CosineLayer, self).__init__(**kwargs)
        self.docs_vector = K.constant(self.docs_encode, dtype="float32")
        self.l2_docs_vector = K.sqrt(K.sum(K.maximum(K.square(self.docs_vector), 1e-12), axis=-1))  # x_inv_norm

    def build(self, input_shape):
        super(CosineLayer, self).build(input_shape)

    def get_config(self):  
        # 防止报错  'NoneType' object has no attribute '_inbound_nodes'
        config = {"docs_vector": self.docs_vector,
                  "l2_docs_vector": self.l2_docs_vector}
        base_config = super(CosineLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, input):
        # 计算余弦相似度
        # square_sum = math_ops.reduce_sum(math_ops.square(x), axis, keepdims=True)
        # x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon))
        # return math_ops.multiply(x, x_inv_norm, name=name)
        #  多了一个 x/sqrt      K.l2_normalize  ===== output = x / sqrt(max(sum(x**2), epsilon))
        
        l2_input = K.sqrt(K.sum(K.maximum(K.square(input), 1e-12), axis=-1))  # x_inv_norm
        fract_0 = K.sum(input * self.docs_vector, axis=-1)
        fract_1 = l2_input * self.l2_docs_vector
        cosine = fract_0 / fract_1
        y_pred_top_k, y_pred_ind_k = tf.nn.top_k(cosine, 10)
        return [y_pred_top_k, y_pred_ind_k]

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[0]]

```


## 再次说明
 - 该方案适合的标准FAQ问答对数量不能太多
 
 
 
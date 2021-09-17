# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/4/15 21:59
# @author  : Mo
# @function: encode of bert-whiteing


from __future__ import print_function, division, absolute_import, division, print_function

# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./."))
sys.path.append(path_root)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(path_root)


from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding
from bert4keras.tokenizers import Tokenizer
from bert4keras.backend import keras, K
from bert4keras.layers import Multiply
from keras.models import Model
import tensorflow as tf

from argparse import Namespace
# from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import json
import time

# shutil.rmtree()


class NonMaskingLayer(keras.layers.Layer):
    """  去除MASK层
    fix convolutional 1D can"t receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


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


class Divide(Multiply):
    """相除
    Divide, Layer that divide a list of inputs.
    
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output /= inputs[i]
        return output


class BertSimModel:
    def __init__(self, config=None):
        """ 初始化超参数、加载预训练模型等 """
        self.config = Namespace(**config)
        self.load_pretrain_model()
        self.eps = 1e-8

    def transform_and_normalize(self, vecs, kernel=None, bias=None):
        """应用变换，然后标准化
        """
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
        norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
        return vecs / np.clip(norms, self.eps, np.inf)

    def compute_kernel_bias(self, vecs):
        """计算kernel和bias
        最后的变换：y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W[:, :self.config.n_components], -mu

    def convert_to_vecs(self, texts):
        """转换文本数据为向量形式
        """

        token_ids = self.convert_to_ids(texts)
        vecs = self.bert_white_encoder.predict(x=[token_ids, np.zeros_like(token_ids)],
                                               batch_size=self.config.batch_size, verbose=self.config.verbose)
        return vecs

    def convert_to_ids(self, texts):
        """转换文本数据为id形式
        """
        token_ids = []
        for text in texts:
            # token_id = self.tokenizer.encode(text, maxlen=self.config.maxlen)[0]
            token_id = self.tokenizer.encode(text, max_length=self.config.maxlen)[0]
            token_ids.append(token_id)
        token_ids = sequence_padding(token_ids)
        return token_ids

    def load_pretrain_model(self):
        """ 加载预训练模型, 和tokenizer """
        self.tokenizer = Tokenizer(os.path.join(self.config.bert_dir, self.config.dict_path), do_lower_case=True)
        # bert-load
        if self.config.pooling == "pooler":
            bert = build_transformer_model(os.path.join(self.config.bert_dir, self.config.config_path),
                                           os.path.join(self.config.bert_dir, self.config.checkpoint_path),
                                           model=self.config.model, with_pool="linear")
        else:
            bert = build_transformer_model(os.path.join(self.config.bert_dir, self.config.config_path),
                                           os.path.join(self.config.bert_dir, self.config.checkpoint_path),
                                           model=self.config.model)
        # output-layers
        outputs, count = [], 0
        while True:
            try:
                output = bert.get_layer("Transformer-%d-FeedForward-Norm" % count).output
                outputs.append(output)
                count += 1
            except:
                break
        # pooling
        if self.config.pooling == "first-last-avg":
            outputs = [NonMaskingLayer()(output_i) for output_i in [outputs[0], outputs[-1]]]
            outputs = [keras.layers.GlobalAveragePooling1D()(fs) for fs in outputs]
            output = keras.layers.Average()(outputs)
        elif self.config.pooling == "first-last-max":
            outputs = [NonMaskingLayer()(output_i) for output_i in [outputs[0], outputs[-1]]]
            outputs = [keras.layers.GlobalMaxPooling1D()(fs) for fs in outputs]
            output = keras.layers.Average()(outputs)
        elif self.config.pooling == "cls-max-avg":
            outputs = [NonMaskingLayer()(output_i) for output_i in [outputs[0], outputs[-1]]]
            outputs_cls = [keras.layers.Lambda(lambda x: x[:, 0])(fs) for fs in outputs]
            outputs_max = [keras.layers.GlobalMaxPooling1D()(fs) for fs in outputs]
            outputs_avg = [keras.layers.GlobalAveragePooling1D()(fs) for fs in outputs]
            output = keras.layers.Concatenate()(outputs_cls + outputs_avg)
        elif self.config.pooling == "last-avg":
            output = keras.layers.GlobalAveragePooling1D()(outputs[-1])
        elif self.config.pooling == "cls-3":
            outputs = [keras.layers.Lambda(lambda x: x[:, 0])(fs) for fs in [outputs[0], outputs[-1], outputs[-2]]]
            output = keras.layers.Concatenate()(outputs)
        elif self.config.pooling == "cls-2":
            outputs = [keras.layers.Lambda(lambda x: x[:, 0])(fs) for fs in [outputs[0], outputs[-1]]]
            output = keras.layers.Concatenate()(outputs)
        elif self.config.pooling == "cls-1":
            output = keras.layers.Lambda(lambda x: x[:, 0])(outputs[-1])
        elif self.config.pooling == "pooler":
            output = bert.output
        # 加载句FAQ标准问的句向量, 并当成一个常量参与余弦相似度的计算
        docs_encode = np.loadtxt(os.path.join(self.config.save_dir, self.config.path_docs_encode))
        # 余弦相似度的层
        score_cosine = CosineLayer(docs_encode)(output)
        # 最后的编码器
        self.bert_white_encoder = Model(bert.inputs, score_cosine)
        print("load bert_white_encoder success!")

    def save_model_builder(self):
        """
        存储为tf-serving的形式
        """
        builder = tf.saved_model.Builder(self.config.path_tfserving)
        signature_def_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.saved_model.build_signature_def(
                # 根据自己模型的要求
                inputs={"Input-Token": tf.saved_model.build_tensor_info(self.bert_white_encoder.input[0]),
                        "Input-Segment": tf.saved_model.build_tensor_info(self.bert_white_encoder.input[1])},
                outputs={"score": tf.saved_model.build_tensor_info(self.bert_white_encoder.output[0]),
                         "doc_id": tf.saved_model.build_tensor_info(self.bert_white_encoder.output[1])},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )}
        builder.add_meta_graph_and_variables(keras.backend.get_session(),  # 注意4
                                             [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=signature_def_map,
                                             # 初始化操作，我的不需要，否则报错
                                             # legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op')
                                             )
        builder.save()

    def train(self, texts):
        """
        训练 
        """
        print("读取文本数:".format(len(texts)))
        print(texts[:3])
        # 文本转成向量vecs
        vecs = self.convert_to_vecs(texts)
        # 训练, 计算变换矩阵和偏置项
        self.config.kernel, self.config.bias = self.compute_kernel_bias(vecs)
        if self.config.ues_white:
            # 生成白化后的句子, 即qa对中的q
            vecs = self.transform_and_normalize(vecs, self.config.kernel, self.config.bias)
        return vecs

    def prob(self, texts):
        """
        编码、白化后的向量 
        """
        vecs_encode = self.convert_to_vecs(texts)
        if self.config.ues_white:
            vecs_encode = self.transform_and_normalize(vecs=vecs_encode, kernel=self.config.kernel, bias=self.config.bias)
        return vecs_encode


if __name__ == '__main__':
    # 存储模型等
    from bertWhiteConf import bert_white_config

    bert_white_model = BertSimModel(bert_white_config)
    bert_white_model.load_pretrain_model()
    bert_white_model.save_model_builder()


    from bertWhiteConf import bert_white_config
    config = Namespace(**bert_white_config)
    tokenizer = Tokenizer(os.path.join(config.bert_dir, config.dict_path), do_lower_case=True)
    text = "你还会什么"
    token_id = tokenizer.encode(text, max_length=config.maxlen)
    print(token_id)


"""
# cpu
docker run -t --rm -p 8532:8501 -v "/TF-SERVING/chatbot_tf:/models/chatbot_tf" -e MODEL_NAME=chatbot_tf tensorflow/serving:latest

# gpu
docker run --runtime=nvidia -p 8532:8501 -v "/TF-SERVING/chatbot_tf:/models/chatbot_tf" -e MODEL_NAME=chatbot_tf tensorflow/serving:1.14.0-gpu

# remarks
batch-size还可以配置batch.cfg等文件

# health testing
curl http://127.0.0.1:8532/v1/models/chatbot_tf

# http test, 不行可以用postman测试
curl -d '{"instances": [{"Input-Token": [2, 870, 6818, 831, 782, 718, 3], "Input-Segment": [0, 0, 0, 0, 0, 0, 0]}]}'   -X  POST  http://localhost:8532/v1/models/chatbot_tf:predict

"""


# python bertWhiteTFServing.py



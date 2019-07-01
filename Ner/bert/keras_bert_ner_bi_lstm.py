# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/5/10 18:05
# @author   :Mo
# @function :classify text of bert and bi-lstm

from __future__ import division, absolute_import

import logging as logger
import numpy as np
import pickle
import codecs

# bert embedding
from Ner.bert.keras_bert_layer import CRF, crf_loss, crf_accuracy
from Ner.bert.keras_bert_embedding import KerasBertEmbedding
from Ner.bert.keras_bert_layer import NonMaskingLayer

# bert trained path
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Bidirectional, CuDNNGRU, CuDNNLSTM
from keras.layers import Dense, Dropout
from keras.layers import GRU, LSTM, TimeDistributed
from keras.objectives import sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers

# bert sequence taging
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_bert import Tokenizer

# corpus path
from Ner.bert import args
from conf.path_config import path_ner_people_train, path_ner_people_dev, path_ner_people_test
from conf.feature_config import vocab_file


class BertNerBiLstmModel():
    def __init__(self):
        # logger.info("BertBiLstmModel init start!")
        print("BertNerBiLstmModel init start!")
        self.dict_path, self.max_seq_len, self.keep_prob, self.is_training = vocab_file, args.max_seq_len, args.keep_prob, args.is_training
        # reader tokenizer
        self.token_dict = {}
        with codecs.open(self.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)

        self.tokenizer = Tokenizer(self.token_dict)
        # 你可以选择一个model build，有bi-lstm single、bi-lstm 3-layers、bi-lstm_attention
        self.build_model_bilstm_layers()
        self.compile_model()
        # self.build_model_bilstm_single()
        # logger.info("BertBiLstmModel init end!")
        print("BertNerBiLstmModel init end!")

    def process_single(self, texts):
        # 文本预处理，传入一个list，返回的是ids\mask\type-ids
        input_ids = []
        input_masks = []
        input_type_ids = []
        for text in texts:
            if type(text) is list:
                text = "".join(text)
            logger.info(text)
            tokens_text = self.tokenizer.tokenize(text)
            logger.info('Tokens:', tokens_text)
            input_id, input_type_id = self.tokenizer.encode(first=text, max_len=self.max_seq_len)
            input_mask = [0 if ids == 0 else 1 for ids in input_id]
            input_ids.append(input_id)
            input_type_ids.append(input_type_id)
            input_masks.append(input_mask)
        # numpy处理list
        input_ids = np.array(input_ids)
        input_masks = np.array(input_masks)
        input_type_ids = np.array(input_type_ids)
        logger.info("process ok!")
        return [input_ids, input_masks, input_type_ids]

    def process_pair(self, textss):
        # 文本预处理，传入一个list，返回的是ids\mask\type-ids
        input_ids = []
        input_masks = []
        input_type_ids = []
        for texts in textss:
            tokens_text = self.tokenizer.tokenize(texts[0])
            logger.info('Tokens1:', tokens_text)
            tokens_text2 = self.tokenizer.tokenize(texts[1])
            logger.info('Tokens2:', tokens_text2)
            input_id, input_type_id = self.tokenizer.encode(first=texts[0], second=texts[1], max_len=self.max_seq_len)
            input_mask = [0 if ids == 0 else 1 for ids in input_id]
            input_ids.append(input_id)
            input_type_ids.append(input_type_id)
            input_masks.append(input_mask)
        # numpy处理list
        input_ids = np.array(input_ids)
        input_masks = np.array(input_masks)
        input_type_ids = np.array(input_type_ids)
        logger.info("process ok!")
        return [input_ids, input_masks, input_type_ids]

    def build_model_bilstm_layers(self):
        if args.use_lstm:
            if args.use_cudnn_cell:
                layer_cell = CuDNNLSTM
            else:
                layer_cell = LSTM
        else:
            if args.use_cudnn_cell:
                layer_cell = CuDNNGRU
            else:
                layer_cell = GRU
        # bert embedding
        bert_inputs, bert_output = KerasBertEmbedding().bert_encode()

        # Bi-LSTM
        x = Bidirectional(layer_cell(units=args.units,
                                     return_sequences=args.return_sequences,
                                     ))(bert_output)
        # 最后
        x = TimeDistributed(Dropout(self.keep_prob))(x)
        dense_layer = Dense(args.max_seq_len, activation=args.activation)(x)
        crf = CRF(args.label, sparse_target=False, learn_mode="join", test_mode='viterbi')
        output_layers = crf(dense_layer)
        self.model = Model(bert_inputs, output_layers)
        self.model.summary(132)

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=args.epsilon, decay=0.0),
            loss=crf_loss if args.use_crf else sparse_categorical_crossentropy,
            metrics=[crf_accuracy] if args.metrics is 'crf_loss' else args.metrics)
        # loss=CRF.loss_function if args.use_crf else categorical_crossentropy,
        # metrics=[CRF.accuracy] if args.metrics is 'crf_loss' else args.metrics)
        # loss=crf.loss if args.use_crf else categorical_crossentropy,
        # metrics=[crf.accuracy] if args.metrics is 'crf_loss' else args.metrics)

    def callback(self):
        cb = [ModelCheckpoint(monitor='val_loss', mode='min', filepath=args.path_save_model, verbose=1, save_best_only=True, save_weights_only=False),
              ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=2, verbose=0, epsilon=1e-6, cooldown=4, min_lr=1e-8),
              EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-8, patience=2)
              ]
        return cb

    def fit(self, x_train, y_train, x_dev, y_dev):
        self.model.fit(x_train, y_train, batch_size=args.batch_size,
                       epochs=args.epochs, validation_data=(x_dev, y_dev),
                       shuffle=True,
                       callbacks=self.callback())
        self.model.save(args.path_save_model)

    def load_model(self):
        print("BertNerBiLstmModel load_model start!")
        # logger.info("BertBiLstmModel load_model start!")
        self.model.load_weights(args.path_save_model)
        # logger.info("BertBiLstmModel load_model end+!")
        print("BertNerBiLstmModel load_model end+!")

    def predict(self, sen):
        input_ids, input_masks, input_type_ids = self.process_single([sen])
        probs = self.model.predict([input_ids, input_masks], batch_size=1)
        probs_first = probs[0]
        preds = []
        for prob_one in probs_first:
            prob_max = np.argmax(prob_one)
            preds.append(prob_max)
        return preds

    def predict_list(self, questions):
        label_preds = []
        for questions_pair in questions:
            input_ids, input_masks, input_type_ids = self.process_single([questions_pair])
            label_pred = self.model.predict([input_ids, input_masks], batch_size=1)
            label_preds.append(label_pred)
        return label_preds


def get_sequence_tagging_data_from_chinese_people_daily_ner_corpus(file_path):
    """
        读取人民日报语料,其实就是一个txt阅读器
    :param file_path: str, text
    :return: list, list
    """
    _x_, _y_ = [], []
    with open(file_path, "r", encoding="utf-8") as fr:
        lines = fr.read().splitlines()
        x, y = [], []
        for line_one in lines:
            rows = line_one.split(" ")
            if len(rows) == 1:
                _x_.append(x), _y_.append(y)
                x, y = [], []
            else:
                x.append(rows[0]), y.append(rows[1])
    return _x_, _y_


def label_tagging(data_x_s, tag_label2index, len_max=32):
    """
        根据类别字典dict、语料y和最大文本长度l，padding和to_categorical
    :param data_x_s: list
    :param tag_label2index:dict 
    :param len_max: int
    :return: list
    """
    tag_labels = []
    for data_x in data_x_s:
        if len(data_x) <= len_max-2:
            tag_labels.append([tag_label2index['O']] + [tag_label2index[i] for i in data_x] + [tag_label2index['O'] for i in range(len_max - len(data_x) - 1)])
        else:
            tag_labels.append([tag_label2index['O']] + [tag_label2index[i] for i in data_x[:len_max-1]] + [tag_label2index['O']])

    tag_labels_pad = pad_sequences(sequences=tag_labels, maxlen=len_max, dtype='int32',
                                padding='post', truncating='post', value=tag_label2index['O'])
    one_hot_y = to_categorical(tag_labels_pad, num_classes=len(tag_label2index))

    label_num = len(set(["".join(str(i)) for i in tag_labels]))
    # tag_labels_pad_to = to_categorical(y=tag_labels_pad.tolist(), num_classes=label_num)
    return one_hot_y, label_num


def label_tagging_predict(y_predicts, tag_i2l):
    y_preds = []
    count_y_predict = y_predicts[0].shape[1]
    for y_predict in y_predicts:
        temp = []
        for i in range(count_y_predict):
            y_predict_list = y_predict[0][i].tolist()
            y_predict_max = y_predict_list.index(max(y_predict_list))
            pred_label = tag_i2l[y_predict_max]
            temp.append(pred_label)
        y_preds.append(temp)
    return y_preds


def create_label_index_dict(data_x_s):
    """
      构建类别和index标签，一一对应等
    :param data_x_s: list, labels of train data 
    :return: list, list
    """
    # 首先构建index2label， 或者label2index
    tag_label2index = {}
    tag_index2label = {}
    data_x_s_one = []
    for d in data_x_s:
        data_x_s_one = data_x_s_one + d
    label_data_x_s = list(set(data_x_s_one))
    for i in range(len(label_data_x_s)):
        tag_label2index[label_data_x_s[i]] = i
        tag_index2label[i] = label_data_x_s[i]
    return tag_label2index, tag_index2label


def process_ner_y(y_data, length_max):
    """
        根据训练语料y生成喂入模型的input_y
    :param y_data: list
    :param length_max: int
    :return: list, dict, dict
    """
    # 保存类别字典
    import os
    if not os.path.exists(args.path_tag_li):
        tag_l2i, tag_i2l = create_label_index_dict(y_data)
        with open(args.path_tag_li, 'wb') as f:
            pickle.dump((tag_l2i, tag_i2l), f)
    else:
        with open(args.path_tag_li, 'rb') as f:
            tag_l2i, tag_i2l = pickle.load(f)
    tagging_index, label_num = y_data, length_max
    try:
        # tagging
        tagging_index, label_num = label_tagging(y_data, tag_l2i, length_max)
    except:
        gg = 0

    return tagging_index, label_num, tag_l2i, tag_i2l


def train():
    # 1. trian
    bert_model = BertNerBiLstmModel()
    # bert_model.compile_model()
    print("process corpus start!")
    # 读取语料
    x_train, y_train = get_sequence_tagging_data_from_chinese_people_daily_ner_corpus(path_ner_people_train)
    x_dev, y_dev = get_sequence_tagging_data_from_chinese_people_daily_ner_corpus(path_ner_people_dev)
    # ques和label index and padding
    x_train = bert_model.process_single(x_train)
    x_dev = bert_model.process_single(x_dev)
    y_train_tagging_index, label_num, tag_l2i, tag_i2l = process_ner_y(y_train, args.max_seq_len)
    y_dev_tagging_index, _,  _, _ = process_ner_y(y_dev, args.max_seq_len)
    # args.label = label_num
    print(label_num)
    print("process corpus end!")
    # gg = x_train[0:2]
    x_train_2 = x_train[0:2]
    x_dev_2 = x_dev[0:2]
    print(x_train_2.__sizeof__())
    print(x_dev_2.__sizeof__())
    y_train_2 = y_train_tagging_index
    y_dev_2 = y_dev_tagging_index

    bert_model.fit(x_train_2, y_train_2, x_dev_2, y_dev_2)


def predict():
    # 3. predict
    _, _, tag_l2i, tag_i2l = process_ner_y([], [])
    bert_model = BertNerBiLstmModel()
    bert_model.load_model()
    pred = bert_model.predict(sen='欧美和台湾经济怎么样')
    tag_labels = []
    for pre in pred:
        tag_labels.append(tag_i2l[pre])
    print(tag_labels)
    while True:
        print("sen: ")
        sen_1 = input()
        pred = bert_model.predict(sen=sen_1)
        tag_labels = []
        for pre in pred:
            tag_labels.append(tag_i2l[pre])
        print(tag_labels)


if __name__ == "__main__":
    train()
    # predict()



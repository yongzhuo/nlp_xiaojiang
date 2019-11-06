# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/5/18 23:51
# @author   :Mo
# @function :classify text of bert and (text-cnn、r-cnn or avt-cnn)

from __future__ import division, absolute_import

from keras.objectives import sparse_categorical_crossentropy, categorical_crossentropy
from conf.path_config import path_webank_train, path_webank_dev, path_webank_test
from keras.layers import Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import SpatialDropout1D, Dropout
from sklearn.metrics import classification_report
from keras.layers import CuDNNGRU, CuDNNLSTM
from keras.layers import Bidirectional
from keras.layers import RepeatVector
from keras.layers import Concatenate
from keras.layers import GRU, LSTM
from keras.layers import Multiply
from keras.layers import Permute
from keras.layers import Lambda
from keras.layers import Dense
from keras.models import Model
from keras import regularizers
import numpy as np
import codecs

import keras.backend as k_keras
import logging as logger

from keras_bert import Tokenizer

from ClassificationText.bert.keras_bert_layer import AttentionWeightedAverage
from ClassificationText.bert.keras_bert_embedding import KerasBertEmbedding
from ClassificationText.bert import args

from conf.feature_config import config_name, ckpt_name, vocab_file, max_seq_len, layer_indexes, gpu_memory_fraction


#from trains import Task
#task = Task.init(project_name="文本相似度", task_name="平安医疗")
def attention(inputs, single_attention_vector=False):
    # attention机制 
    time_steps = k_keras.int_shape(inputs)[1]
    input_dim = k_keras.int_shape(inputs)[2]
    x = Permute((2, 1))(inputs)
    x = Dense(time_steps, activation='softmax')(x)
    if single_attention_vector:
        x = Lambda(lambda x: k_keras.mean(x, axis=1))(x)
        x = RepeatVector(input_dim)(x)

    a_probs = Permute((2, 1))(x)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


class BertTextCnnModel():
    def __init__(self):
        # logger.info("BertBiLstmModel init start!")
        print("BertBiLstmModel init start!")
        self.config_path, self.checkpoint_path, self.dict_path = config_name, ckpt_name, vocab_file
        self.max_seq_len, self.filters, self.embedding_dim, self.keep_prob = args.max_seq_len, args.filters, args.embedding_dim, args.keep_prob
        self.activation, self.label = args.activation, args.label
        # reader tokenizer
        self.token_dict = {}
        with codecs.open(self.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)

        self.tokenizer = Tokenizer(self.token_dict)
        # 这里模型可以选text-rnn、r-cnn或者是avt-cnn
        # self.build_model_text_cnn()
        # self.build_model_r_cnn()
        self.build_model_avt_cnn()
        # logger.info("BertBiLstmModel init end!")
        print("BertBiLstmModel init end!")

    def build_model_text_cnn(self):
        #########    text-cnn    #########
        # bert embedding
        bert_inputs, bert_output = KerasBertEmbedding().bert_encode()
        # text cnn
        bert_output_emmbed = SpatialDropout1D(rate=self.keep_prob)(bert_output)
        concat_out = []
        for index, filter_size in enumerate(self.filters):
            x = Conv1D(name='TextCNN_Conv1D_{}'.format(index), filters=int(self.embedding_dim/2), kernel_size=self.filters[index], padding='valid', kernel_initializer='normal', activation='relu')(bert_output_emmbed)
            x = GlobalMaxPooling1D(name='TextCNN_MaxPool1D_{}'.format(index))(x)
            concat_out.append(x)
        x = Concatenate(axis=1)(concat_out)
        x = Dropout(self.keep_prob)(x)

        # 最后就是softmax
        dense_layer = Dense(self.label, activation=self.activation)(x)
        output_layers = [dense_layer]
        self.model = Model(bert_inputs, output_layers)

    def build_model_r_cnn(self):
        #########    RCNN    #########
        # bert embedding
        bert_inputs, bert_output = KerasBertEmbedding().bert_encode()
        # rcnn
        bert_output_emmbed = SpatialDropout1D(rate=self.keep_prob)(bert_output)
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

        x = Bidirectional(layer_cell(units=args.units, return_sequences=args.return_sequences,
                                     kernel_regularizer=regularizers.l2(args.l2 * 0.1),
                                     recurrent_regularizer=regularizers.l2(args.l2)
                                     ))(bert_output_emmbed)
        x = Dropout(args.keep_prob)(x)
        x = Conv1D(filters=int(self.embedding_dim / 2), kernel_size=2, padding='valid', kernel_initializer='normal', activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(args.keep_prob)(x)
        # 最后就是softmax
        dense_layer = Dense(self.label, activation=self.activation)(x)
        output_layers = [dense_layer]
        self.model = Model(bert_inputs, output_layers)

    def build_model_avt_cnn(self):
        #########text-cnn#########
        # bert embedding
        bert_inputs, bert_output = KerasBertEmbedding().bert_encode()
        # text cnn
        bert_output_emmbed = SpatialDropout1D(rate=self.keep_prob)(bert_output)
        concat_x = []
        concat_y = []
        concat_z = []
        for index, filter_size in enumerate(self.filters):
            conv = Conv1D(name='TextCNN_Conv1D_{}'.format(index), filters=int(self.embedding_dim/2), kernel_size=self.filters[index], padding='valid', kernel_initializer='normal', activation='relu')(bert_output_emmbed)
            x = GlobalMaxPooling1D(name='TextCNN_MaxPooling1D_{}'.format(index))(conv)
            y = GlobalAveragePooling1D(name='TextCNN_AveragePooling1D_{}'.format(index))(conv)
            z = AttentionWeightedAverage(name='TextCNN_Annention_{}'.format(index))(conv)
            concat_x.append(x)
            concat_y.append(y)
            concat_z.append(z)

        merge_x = Concatenate(axis=1)(concat_x)
        merge_y = Concatenate(axis=1)(concat_y)
        merge_z = Concatenate(axis=1)(concat_z)
        merge_xyz = Concatenate(axis=1)([merge_x, merge_y, merge_z])
        x = Dropout(self.keep_prob)(merge_xyz)

        # 最后就是softmax
        dense_layer = Dense(self.label, activation=self.activation)(x)
        output_layers = [dense_layer]
        self.model = Model(bert_inputs, output_layers)

    def compile_model(self):
        self.model.compile(optimizer=args.optimizers,
                           loss=categorical_crossentropy,
                           metrics=args.metrics)

    def callback(self):
        c_b = [ModelCheckpoint(args.path_save_model, monitor='val_loss', verbose=1, save_best_only=True,
                               save_weights_only=False, mode='min'),
               EarlyStopping(min_delta=1e-9, patience=4, mode='min')
               ]
        return c_b

    def fit(self, x_train, y_train, x_dev, y_dev):
        self.model.fit(x_train, y_train, batch_size=args.batch_size,
                       epochs=args.epochs, validation_data=(x_dev, y_dev),
                       shuffle=True,
                       callbacks=self.callback())
        self.model.save(args.path_save_model)

    def load_model(self):
        print("BertBiLstmModel load_model start!")
        # logger.info("BertBiLstmModel load_model start!")
        self.model.load_weights(args.path_save_model)
        # logger.info("BertBiLstmModel load_model end+!")
        print("BertBiLstmModel load_model end+!")

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
        return input_ids, input_masks, input_type_ids

    def predict(self, sen_1, sen_2):
        input_ids, input_masks, input_type_ids = self.process_pair([[sen_1, sen_2]])
        return self.model.predict([input_ids, input_masks], batch_size=1)

    def predict_list(self, questions):
        label_preds = []
        for questions_pair in questions:
            input_ids, input_masks, input_type_ids = self.process_pair([questions_pair])
            label_pred = self.model.predict([input_ids, input_masks], batch_size=1)
            label_preds.append(label_pred[0])
        return label_preds


def classify_pair_corpus_webank(bert_model, path_webank):
    # 数据预处理
    from utils.text_tools import text_preprocess, txtRead, txtWrite
    import random

    webank_q_2_l = txtRead(path_webank, encodeType='utf-8')
    questions = []
    labels = []
    for ques_label in webank_q_2_l[1:]:
        q_2_l = ques_label.split(',')
        q_1 = q_2_l[0]
        q_2 = "".join(q_2_l[1:-1])
        label = q_2_l[-1]
        questions.append([text_preprocess(q_1), text_preprocess(q_2)])
        label_int = int(label)
        label_arr = [0] * args.label
        label_arr[label_int] = 1
        labels.append(label_arr)

    questions = np.array(questions)
    labels = np.array(labels)

    input_ids, input_masks, input_type_ids = bert_model.process_pair(questions)

    return questions, labels, input_ids, input_masks, input_type_ids


def train():
    # 1. trian
    bert_model = BertTextCnnModel()
    bert_model.compile_model()
    _, labels_train, input_ids_train, input_masks_train, _ = classify_pair_corpus_webank(bert_model, path_webank_train)
    _, labels_dev, input_ids_dev, input_masks_dev, _ = classify_pair_corpus_webank(bert_model, path_webank_dev)
    # questions_test, labels_test, input_ids_test, input_masks_test, _ = classify_pair_corpus_webank(bert_model, path_webank_test)
    print("process corpus ok!")
    bert_model.fit([input_ids_train, input_masks_train], labels_train, [input_ids_dev, input_masks_dev], labels_dev)
    print("bert_model fit ok!")


def tet():
    #  2.test
    bert_model = BertTextCnnModel()
    bert_model.load_model()
    questions_test, labels_test, input_ids_test, input_masks_test, _ = classify_pair_corpus_webank(bert_model,
                                                                                                   path_webank_test)
    print('predict_list start! you will wait for a few minutes')
    labels_pred = bert_model.predict_list(questions_test)
    print('predict_list end!')

    labels_pred_np = np.array(labels_pred)
    labels_pred_np_arg = np.argmax(labels_pred_np, axis=1)
    labels_test_np = np.array(labels_test)
    labels_test_np_arg = np.argmax(labels_test_np, axis=1)
    #target_names = ['不相似', '相似'] 
    #report_predict = classification_report(labels_test_np_arg, labels_pred_np_arg, target_names=target_names, digits=9)
    report_predict = classification_report(labels_test_np_arg, labels_pred_np_arg, digits=9)
    print(report_predict)


def predict():
    # 3. predict
    bert_model = BertTextCnnModel()
    bert_model.load_model()
    pred = bert_model.predict(sen_1='jy', sen_2='myz')
    print(pred[0])
    while True:
        print("sen_1: ")
        sen_1 = input()
        print("sen_2: ")
        sen_2 = input()
        pred = bert_model.predict(sen_1=sen_1, sen_2=sen_2)
        lable = np.argmax(pred)
        print(lable)


if __name__ == "__main__":
    #train()
    #tet()
    predict()

# text cnn, real stop
# 100000/100000 [==============================] - 1546s 15ms/step - loss: 0.4168 - acc: 0.8108 - val_loss: 0.4379 - val_acc: 0.8008
# Epoch 00024: val_loss improved from 0.43926 to 0.43788, saving model to model_webank_tdt/bert_avt_cnn.h5
#                  precision    recall  f1-score   support
#         不相似  0.800245600 0.782000000 0.791017601      5000
#          相似  0.786859601 0.804800000 0.795728693      5000
# avg / total  0.793552600 0.793400000 0.793373147     10000


# text-rcnn, real stop
# 100000/100000 [==============================] - 1671s 17ms/step - loss: 0.4627 - acc: 0.7971 - val_loss: 0.4810 - val_acc: 0.8018
#                    precision    recall  f1-score   support
#         不相似  0.777479378 0.810600000 0.793694311      5000
#          相似  0.802172551 0.768000000 0.784714417      5000
# avg / total  0.789825965 0.789300000 0.789204364     10000


# avt-cnn, real stop
# 100000/100000 [==============================] - 1562s 16ms/step - loss: 0.4204 - acc: 0.8091 - val_loss: 0.4391 - val_acc: 0.7925
# Epoch 00015: val_loss improved from 0.44410 to 0.43914, saving model to model_webank_tdt/bert_avt_cnn.h5
#                   precision    recall  f1-score   support
#         不相似  0.789808917 0.768800000 0.779162866      5000
#          相似  0.774790571 0.795400000 0.784960032      5000
# avg / total  0.782299744 0.782100000 0.782061449     10000

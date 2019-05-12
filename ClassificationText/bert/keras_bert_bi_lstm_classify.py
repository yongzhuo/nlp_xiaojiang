# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/5/10 18:05
# @author   :Mo
# @function :classify text of bert and bi-lstm

from __future__ import division, absolute_import

from keras.objectives import sparse_categorical_crossentropy, categorical_crossentropy
from conf.path_config import path_webank_train, path_webank_dev, path_webank_test
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import GlobalAvgPool1D, GlobalMaxPool1D
from sklearn.metrics import classification_report
from keras.layers import CuDNNGRU, CuDNNLSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import concatenate
from keras.layers import GRU, LSTM
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Dense
from keras.models import Model
from keras import regularizers
import numpy as np
import codecs
import keras

from ClassificationText.bert.keras_bert_embedding import KerasBertEmbedding
from ClassificationText.bert import args

from conf.feature_config import config_name, ckpt_name, vocab_file, max_seq_len, layer_indexes, gpu_memory_fraction
from keras_bert import Tokenizer

import logging as logger


class BertBiLstmModel():
    def __init__(self):
        # logger.info("BertBiLstmModel init start!")
        print("BertBiLstmModel init start!")
        self.config_path, self.checkpoint_path, self.dict_path, self.max_seq_len = config_name, ckpt_name, vocab_file, max_seq_len
        # reader tokenizer
        self.token_dict = {}
        with codecs.open(self.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)

        self.tokenizer = Tokenizer(self.token_dict)
        self.build_model()
        # logger.info("BertBiLstmModel init end!")
        print("BertBiLstmModel init end!")

    def process_single(self, texts):
        # 文本预处理，传入一个list，返回的是ids\mask\type-ids
        input_ids = []
        input_masks = []
        input_type_ids = []
        for text in texts:
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
        return input_ids, input_masks, input_type_ids

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

    def build_model(self):
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
        # bert_output = bert_output[:0:]
        # layer_get_cls = Lambda(lambda x: x[:, 0:1, :])
        # bert_output = layer_get_cls(bert_output)
        # print("layer_get_cls:")
        # print(bert_output.shape)
        # Bi-LSTM
        x = Bidirectional(layer_cell(units=args.units, return_sequences=args.return_sequences,
                                               kernel_regularizer=regularizers.l2(args.l2*0.1),
                                               recurrent_regularizer=regularizers.l2(args.l2)
                                               ))(bert_output)
        # blstm_layer = TimeDistributed(Dropout(args.keep_prob))(blstm_layer) 这个用不了，好像是输入不对, dims<3吧
        x = Dropout(args.keep_prob)(x)

        x = Bidirectional(layer_cell(units=args.units, return_sequences=args.return_sequences,
                                               kernel_regularizer=regularizers.l2(args.l2*0.1),
                                               recurrent_regularizer=regularizers.l2(args.l2)))(x)
        x = Dropout(args.keep_prob)(x)
        x = Bidirectional(layer_cell(units=args.units, return_sequences=args.return_sequences,
                                               kernel_regularizer=regularizers.l2(args.l2*0.1),
                                               recurrent_regularizer=regularizers.l2(args.l2)))(x)
        x = Dropout(args.keep_prob)(x)

        # 平均池化、最大池化拼接
        avg_pool = GlobalAvgPool1D()(x)
        max_pool = GlobalMaxPool1D()(x)
        print(max_pool.shape)
        print(avg_pool.shape)
        concat = concatenate([avg_pool, max_pool])
        x = Dense(int(args.units/4), activation="relu")(concat)
        x = Dropout(args.keep_prob)(x)

        # 最后就是softmax
        dense_layer = Dense(args.label, activation=args.activation)(x)
        output_layers = [dense_layer]
        self.model = Model(bert_inputs, output_layers)

    def build_model_bilstm_single(self):
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
        x = Bidirectional(layer_cell(units=args.units, return_sequences=args.return_sequences,
                                               kernel_regularizer=regularizers.l2(args.l2*0.1),
                                               recurrent_regularizer=regularizers.l2(args.l2)
                                               ))(bert_output)
        x = Dropout(args.keep_prob)(x)

        # 最后就是softmax
        dense_layer = Dense(args.label, activation=args.activation)(x)
        output_layers = [dense_layer]
        self.model = Model(bert_inputs, output_layers)

    def compile_model(self):
        self.model.compile(optimizer=args.optimizers,
                           loss=categorical_crossentropy,
                           metrics=args.metrics)

    def callback(self):
        cb = [ModelCheckpoint(args.path_save_model, monitor='val_loss',
                              verbose=1, save_best_only=True, save_weights_only=False, mode='min'),
              EarlyStopping(min_delta=1e-8, patience=10, mode='min'),
              ReduceLROnPlateau(factor=0.2, patience=6, verbose=0, mode='min', epsilon=1e-6, cooldown=4, min_lr=1e-8)
              ]
        return cb

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


def classify_single_corpus(bert_model):
    # 数据预处理
    from utils.text_tools import text_preprocess, txtRead, txtWrite
    from conf.path_config import path_webank_sim
    import random

    webank_q_2_l = txtRead(path_webank_sim, encodeType='gbk')
    questions = []
    labels = []
    for ques_label in webank_q_2_l[1:]:
        q_2_l = ques_label.split(',')
        q_2 = "".join(q_2_l[:-1])
        label = q_2_l[-1]
        questions.append(text_preprocess(q_2))
        label_int = int(label)
        labels.append([0, 1] if label_int == 1 else [1, 0])

    questions = np.array(questions)
    labels = np.array(labels)
    index = [i for i in range(len(labels))]
    random.shuffle(index)
    questions = questions[index]
    labels = labels[index]
    len_train = int(len(labels) * 0.9)

    train_x, train_y = questions[0:len_train], labels[0:len_train]
    test_x, test_y = questions[len_train:], labels[len_train:]

    input_ids, input_masks, input_type_ids = bert_model.process_single(train_x)
    input_ids2, input_masks2, input_type_ids2 = bert_model.process_single(test_x)

    return train_x, train_y, test_x, test_y, input_ids, input_masks, input_type_ids, input_ids2, input_masks2, input_type_ids2


def classify_pair_corpus(bert_model):
    # 数据预处理
    from utils.text_tools import text_preprocess, txtRead, txtWrite
    from conf.path_config import path_webank_sim
    import random

    webank_q_2_l = txtRead(path_webank_sim, encodeType='gbk')
    questions = []
    labels = []
    for ques_label in webank_q_2_l[1:]:
        q_2_l = ques_label.split(',')
        q_1 = q_2_l[0]
        q_2 = "".join(q_2_l[1:-1])
        label = q_2_l[-1]
        questions.append([text_preprocess(q_1), text_preprocess(q_2)])
        label_int = int(label)
        labels.append([0, 1] if label_int==1 else [1, 0] )

    questions = np.array(questions)
    labels = np.array(labels)
    index = [i for i in range(len(labels))]
    random.shuffle(index)
    questions = questions[index]
    labels = labels[index]
    len_train = int(len(labels) * 0.9)

    train_x, train_y = questions[0:len_train], labels[0:len_train]
    test_x, test_y = questions[len_train:], labels[len_train:]

    input_ids, input_masks, input_type_ids = bert_model.process_pair(train_x)
    input_ids2, input_masks2, input_type_ids2 = bert_model.process_pair(test_x)

    return train_x, train_y, test_x, test_y, input_ids, input_masks, input_type_ids, input_ids2, input_masks2, input_type_ids2


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
        labels.append([0, 1] if label_int==1 else [1, 0] )

    questions = np.array(questions)
    labels = np.array(labels)

    input_ids, input_masks, input_type_ids = bert_model.process_pair(questions)

    return questions, labels, input_ids, input_masks, input_type_ids


if __name__=="__main__":
    # 1. trian
    bert_model = BertBiLstmModel()
    bert_model.compile_model()
    _, labels_train, input_ids_train, input_masks_train, _ = classify_pair_corpus_webank(bert_model, path_webank_train)
    _, labels_dev, input_ids_dev, input_masks_dev, _ = classify_pair_corpus_webank(bert_model, path_webank_dev)
    # questions_test, labels_test, input_ids_test, input_masks_test, _ = classify_pair_corpus_webank(bert_model, path_webank_test)
    print("process corpus ok!")
    bert_model.fit([input_ids_train, input_masks_train], labels_train, [input_ids_dev, input_masks_dev], labels_dev)
    print("bert_model fit ok!")


    # #  2.test
    # bert_model = BertBiLstmModel()
    # bert_model.load_model()
    # questions_test, labels_test, input_ids_test, input_masks_test, _ = classify_pair_corpus_webank(bert_model, path_webank_test)
    # print('predict_list start! you will wait for a few minutes')
    # labels_pred = bert_model.predict_list(questions_test)
    # print('predict_list end!')
    #
    # labels_pred_np = np.array(labels_pred)
    # labels_pred_np_arg = np.argmax(labels_pred_np, axis=1)
    # labels_test_np = np.array(labels_test)
    # labels_test_np_arg = np.argmax(labels_test_np, axis=1)
    # target_names = ['不相似', '相似']
    # report_predict = classification_report(labels_test_np_arg, labels_pred_np_arg, target_names=target_names)
    # print(report_predict)


    # # 3. predict
    # bert_model = BertBiLstmModel()
    # bert_model.load_model()
    # pred = bert_model.predict(sen_1='jy', sen_2='myz')
    # print(pred[0][1])
    # while True:
    #     print("sen_1: ")
    #     sen_1 = input()
    #     print("sen_2: ")
    #     sen_2 = input()
    #     pred = bert_model.predict(sen_1=sen_1, sen_2=sen_2)
    #     print(pred[0][1])


    # # classify single webank and ali
    # train_x, train_y, test_x, test_y, input_ids, input_masks, input_type_ids, input_ids2, input_masks2, input_type_ids2 = classify_single_corpus(bert_model)
    # # classify pair
    # train_x, train_y, test_x, test_y, input_ids, input_masks, input_type_ids, input_ids2, input_masks2, input_type_ids2 = classify_pair_corpus(bert_model)


# result of train 100/31 all spare
# 18017/18017 [==============================] - 110s 6ms/step - loss: 0.2321 - acc: 0.9079 - val_loss: 0.5854 - val_acc: 0.7827
# result of train 100/100 cls
# 18017/18017 [==============================] - 82s 5ms/step - loss: 0.5843 - acc: 0.6938 - val_loss: 0.5537 - val_acc: 0.7093
# result of train 100/24 1 lstm single
# 18017/18017 [==============================] - 95s 5ms/step - loss: 0.4559 - acc: 0.8015 - val_loss: 0.5184 - val_acc: 0.7682
# result of train 100/11 3 lstm MaxPool1D
# 18017/18017 [==============================] - 91s 5ms/step - loss: 0.2837 - acc: 0.8863 - val_loss: 0.4547 - val_acc: 0.7982
# result of train 100/8 3  lstm AvgPool1D
# 18017/18017 [==============================] - 91s 5ms/step - loss: 0.3544 - acc: 0.8460 - val_loss: 0.4707 - val_acc: 0.7897
# result of train 100/7 3  lstm MaxPool1D、AvgPool1D
# 18017/18017 [==============================] - 89s 5ms/step - loss: 0.4505 - acc: 0.8031 - val_loss: 0.4728 - val_acc: 0.7742


# train data of tdt 0.78
# test {0: {'precision': 0.757, 'recall': 0.804, 'f1': 0.779}, 1: {'precision': 0.791, 'recall': 0.741, 'f1': 0.765}, 'mean': {'mean_precision': 0.774, 'mean_recall': 0.772, 'macro_f1': 0.772}, 'sum': {'sum_precision': 0.773, 'sum_recall': 0.749, 'micro_f1': 0.761}}
# train data of tdt avgpool
# test {0: {'precision': 0.773, 'recall': 0.781, 'f1': 0.777}, 1: {'precision': 0.779, 'recall': 0.771, 'f1': 0.775}, 'mean': {'mean_precision': 0.776, 'mean_recall': 0.776, 'macro_f1': 0.776}, 'sum': {'sum_precision': 0.776, 'sum_recall': 0.772, 'micro_f1': 0.774}}


# test data of tdt; result of train 100/2 3 lstm MaxPool1D、AvgPool1D
#                  precision    recall  f1-score   support
#         不相似       0.78      0.77      0.77      5000
#          相似       0.77      0.78      0.78      5000
# avg / total       0.78      0.78      0.78     10000
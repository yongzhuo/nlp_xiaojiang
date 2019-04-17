"""
对SequenceToSequence模型进行基本的参数组合测试
"""

from utils.mode_util.seq2seq.thread_generator import ThreadedGenerator
from utils.mode_util.seq2seq.model_seq2seq import SequenceToSequence
from utils.mode_util.seq2seq.word_sequence import WordSequence
from utils.mode_util.seq2seq.data_utils import batch_flow

from conf.path_config import model_ckpt_web_anti_word
from conf.path_config import train_data_web_xyw_anti
from conf.path_config import train_data_web_emb_anti
from conf.path_config import path_webank_sim
from conf.path_config import path_params

import tensorflow as tf
import numpy as np
import random
import pickle
import jieba
import sys


sys.path.append('..')


def pred_word_anti(bidirectional, cell_type, depth,
         attention_type, use_residual, use_dropout, time_major, hidden_units):
    """测试不同参数在生成的假数据上的运行结果"""

    x_data, _, ws = pickle.load(open(train_data_web_xyw_anti, 'rb'))

    for x in x_data[:5]:
        print(' '.join(x))

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = model_ckpt_web_anti_word

    # 测试部分
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=1,
        bidirectional=bidirectional,
        cell_type=cell_type,
        depth=depth,
        attention_type=attention_type,
        use_residual=use_residual,
        use_dropout=use_dropout,
        parallel_iterations=1,
        time_major=time_major,
        hidden_units=hidden_units,
        share_embedding=True,
        pretrained_embedding=True
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            user_text = input('Input Chat Sentence:')
            if user_text in ('exit', 'quit'):
                exit(0)
            x_test = [jieba.lcut(user_text.lower())]
            # x_test = [word_tokenize(user_text)]
            bar = batch_flow([x_test], ws, 1)
            x, xl = next(bar)
            x = np.flip(x, axis=1)
            # x = np.array([
            #     list(reversed(xx))
            #     for xx in x
            # ])
            print(x, xl)
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(pred)
            # prob = np.exp(prob.transpose())
            print(ws.inverse_transform(x[0]))
            # print(ws.inverse_transform(pred[0]))
            # print(pred.shape, prob.shape)
            for p in pred:
                ans = ws.inverse_transform(p)
                print(ans)


def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    pred_word_anti(
        bidirectional=True,
        cell_type='lstm',
        depth=2,
        attention_type='Bahdanau',
        use_residual=False,
        use_dropout=False,
        time_major=False,
        hidden_units=512
    )


if __name__ == '__main__':
    main()

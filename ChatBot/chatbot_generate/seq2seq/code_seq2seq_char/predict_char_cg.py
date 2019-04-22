"""
对SequenceToSequence模型进行基本的参数组合测试
Code from: QHDuan(2018-02-05) url: https://github.com/qhduan/just_another_seq2seq

"""

from utils.mode_util.seq2seq.data_utils import batch_flow_bucket as batch_flow
from utils.mode_util.seq2seq.thread_generator import ThreadedGenerator
from utils.mode_util.seq2seq.model_seq2seq import SequenceToSequence
from utils.mode_util.seq2seq.word_sequence import WordSequence

from conf.path_config import chicken_and_gossip_path
from conf.path_config import chatbot_data_cg_char_dir
from conf.path_config import chatbot_data_cg_ws_anti
from conf.path_config import chatbot_data_cg_xy_anti
from conf.path_config import model_ckpt_cg_anti
from conf.path_config import path_params

import tensorflow as tf
import numpy as np
import pickle
import json
import sys

sys.path.append('..')


def predict_anti(params):
    """测试不同参数在生成的假数据上的运行结果"""

    x_data, _ = pickle.load(open(chatbot_data_cg_xy_anti, 'rb'))
    ws = pickle.load(open(chatbot_data_cg_ws_anti, 'rb'))

    for x in x_data[:5]:
        print(' '.join(x))

    config = tf.ConfigProto(
        # device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = model_ckpt_cg_anti

    # 测试部分
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=0,
        **params
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            user_text = input('Input Chat Sentence:')
            if user_text in ('exit', 'quit'):
                exit(0)
            x_test = [list(user_text.lower())]
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
    """入口程序"""
    import json
    predict_anti(json.load(open(path_params)))


if __name__ == '__main__':
    main()
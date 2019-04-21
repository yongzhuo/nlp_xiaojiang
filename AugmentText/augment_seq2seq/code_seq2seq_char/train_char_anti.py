"""
对SequenceToSequence模型进行基本的参数组合测试
Code from: QHDuan(2018-02-05) url: https://github.com/qhduan/just_another_seq2seq

"""

from utils.mode_util.seq2seq.data_utils import batch_flow_bucket as batch_flow
from utils.mode_util.seq2seq.thread_generator import ThreadedGenerator
from utils.mode_util.seq2seq.model_seq2seq import SequenceToSequence
from utils.mode_util.seq2seq.word_sequence import WordSequence

from conf.path_config import train_data_web_ws_anti
from conf.path_config import train_data_web_xy_anti
from conf.path_config import model_ckpt_web_anti
from conf.path_config import path_params

from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pickle
import random
import json
import sys

sys.path.append('..')


def train_and_dev(params):
    """测试不同参数在生成的假数据上的运行结果"""

    x_data, y_data = pickle.load(open(train_data_web_xy_anti, 'rb'))
    ws = pickle.load(open(train_data_web_ws_anti, 'rb'))

    # 训练部分
    n_epoch = 2
    batch_size = 128
    x_data, y_data = shuffle(x_data, y_data, random_state=20190412)

    steps = int(len(x_data) / batch_size) + 1

    config = tf.ConfigProto(
        # device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = model_ckpt_web_anti

    tf.reset_default_graph()
    with tf.Graph().as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        with tf.Session(config=config) as sess:

            model = SequenceToSequence(
                input_vocab_size=len(ws),
                target_vocab_size=len(ws),
                batch_size=batch_size,
                **params
            )
            init = tf.global_variables_initializer()
            sess.run(init)


            flow = ThreadedGenerator(
                batch_flow([x_data, y_data], ws, batch_size,
                           add_end=[False, True]),
                queue_maxsize=30)

            dummy_encoder_inputs = np.array([
                np.array([WordSequence.PAD]) for _ in range(batch_size)])
            dummy_encoder_inputs_lengths = np.array([1] * batch_size)

            for epoch in range(1, n_epoch + 1):
                costs = []
                bar = tqdm(range(steps), total=steps,
                           desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    x, xl, y, yl = next(flow)
                    x = np.flip(x, axis=1)

#                     add_loss = model.train(sess,
#                                            dummy_encoder_inputs,
#                                            dummy_encoder_inputs_lengths,
#                                            y, yl, loss_only=True)

#                     add_loss *= -0.5
                    
                    
                    loss_aq = model.train(sess,
                                          y, yl,
                                          x, xl,
                                          loss_only=True)

                    add_loss = 0.1 * loss_aq

                    cost, lr = model.train(sess, x, xl, y, yl,
                                           return_lr=True, add_loss=add_loss)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f} lr={:.6f}'.format(
                        epoch,
                        np.mean(costs),
                        lr
                    ))

                model.save(sess, save_path)

            flow.close()

    # 测试部分
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=12,
        **params
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        bar = batch_flow([x_data, y_data], ws, 1, add_end=False)
        t = 0
        for x, xl, y, yl in bar:
            x = np.flip(x, axis=1)
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(ws.inverse_transform(x[0]))
            print(ws.inverse_transform(y[0]))
            print(ws.inverse_transform(pred[0]))
            t += 1
            if t >= 3:
                break

    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=1,
        **params
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        bar = batch_flow([x_data, y_data], ws, 1, add_end=False)
        t = 0
        for x, xl, y, yl in bar:
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(ws.inverse_transform(x[0]))
            print(ws.inverse_transform(y[0]))
            print(ws.inverse_transform(pred[0]))
            t += 1
            if t >= 3:
                break


def main():
    """入口程序"""
    train_and_dev(json.load(open(path_params)))


if __name__ == '__main__':
    main()

"""
对SequenceToSequence模型进行基本的参数组合测试
"""

from utils.mode_util.seq2seq.data_utils import batch_flow
from utils.mode_util.seq2seq.thread_generator import ThreadedGenerator
from utils.mode_util.seq2seq.model_seq2seq import SequenceToSequence
from utils.mode_util.seq2seq.word_sequence import WordSequence

from conf.path_config import model_ckpt_web_anti_word
from conf.path_config import train_data_web_xyw_anti
from conf.path_config import train_data_web_emb_anti
from conf.path_config import path_webank_sim
from conf.path_config import path_params

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random
import pickle
import sys

sys.path.append('..')


def train_word_anti(bidirectional, cell_type, depth,
         attention_type, use_residual, use_dropout, time_major, hidden_units):
    """测试不同参数在生成的假数据上的运行结果"""

    emb = pickle.load(open(train_data_web_emb_anti, 'rb'))

    x_data, y_data, ws = pickle.load(
        open(train_data_web_xyw_anti, 'rb'))

    # 训练部分
    n_epoch = 10
    batch_size = 128
    # x_data, y_data = shuffle(x_data, y_data, random_state=0)
    # x_data = x_data[:100000]
    # y_data = y_data[:100000]
    steps = int(len(x_data) / batch_size) + 1

    config = tf.ConfigProto(
        # device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = model_ckpt_web_anti_word

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
                bidirectional=bidirectional,
                cell_type=cell_type,
                depth=depth,
                attention_type=attention_type,
                use_residual=use_residual,
                use_dropout=use_dropout,
                hidden_units=hidden_units,
                time_major=time_major,
                learning_rate=0.001,
                optimizer='adam',
                share_embedding=True,
                dropout=0.2,
                pretrained_embedding=True
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            # 加载训练好的embedding
            model.feed_embedding(sess, encoder=emb)

            # print(sess.run(model.input_layer.kernel))
            # exit(1)

            flow = ThreadedGenerator(
                batch_flow([x_data, y_data], ws, batch_size),
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

                    add_loss = model.train(sess,
                                           dummy_encoder_inputs,
                                           dummy_encoder_inputs_lengths,
                                           y, yl, loss_only=True)

                    add_loss *= -0.5
                    # print(x, y)
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
        bidirectional=bidirectional,
        cell_type=cell_type,
        depth=depth,
        attention_type=attention_type,
        use_residual=use_residual,
        use_dropout=use_dropout,
        hidden_units=hidden_units,
        time_major=time_major,
        parallel_iterations=1,
        learning_rate=0.001,
        optimizer='adam',
        share_embedding=True,
        pretrained_embedding=True
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        bar = batch_flow([x_data, y_data], ws, 1)
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
        bidirectional=bidirectional,
        cell_type=cell_type,
        depth=depth,
        attention_type=attention_type,
        use_residual=use_residual,
        use_dropout=use_dropout,
        hidden_units=hidden_units,
        time_major=time_major,
        parallel_iterations=1,
        learning_rate=0.001,
        optimizer='adam',
        share_embedding=True,
        pretrained_embedding=True
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        bar = batch_flow([x_data, y_data], ws, 1)
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
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    train_word_anti(
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

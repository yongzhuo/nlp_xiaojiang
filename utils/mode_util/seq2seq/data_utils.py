"""
一些数据操作所需的模块
Code from: QHDuan(2018-02-05) url: https://github.com/qhduan/just_another_seq2seq

"""

from utils.mode_util.seq2seq.word_sequence import WordSequence
from tensorflow.python.client import device_lib
import numpy as np
import random


VOCAB_SIZE_THRESHOLD_CPU = 50000


def _get_available_gpus():
    """获取当前可用GPU数量"""
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def _get_embed_device(vocab_size):
    """Decide on which device to place an embed matrix given its vocab size.
    根据输入输出的字典大小，选择在CPU还是GPU上初始化embedding向量
    """
    gpus = _get_available_gpus()
    if not gpus or vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    return "/gpu:0"


def transform_sentence(sentence, ws, max_len=None, add_end=False):
    """转换一个单独句子
    Args:
        sentence: 一句话，例如一个数组['你', '好', '吗']
        ws: 一个WordSequence对象，转换器
        max_len:
            进行padding的长度，也就是如果sentence长度小于max_len
            则padding到max_len这么长
    Ret:
        encoded:
            一个经过ws转换的数组，例如[4, 5, 6, 3]
        encoded_len: 上面的长度
    """
    encoded = ws.transform(
        sentence,
        max_len=max_len if max_len is not None else len(sentence))
    encoded_len = len(sentence) + (1 if add_end else 0) # add end
    if encoded_len > len(encoded):
        encoded_len = len(encoded)
    return encoded, encoded_len


def batch_flow(data, ws, batch_size, raw=False, add_end=True):
    """从数据中随机 batch_size 个的数据，然后 yield 出去
    Args:
        data:
            是一个数组，必须包含一个护着更多个同等的数据队列数组
        ws:
            可以是一个WordSequence对象，也可以是多个组成的数组
            如果是多个，那么数组数量应该与data的数据数量保持一致，即len(data) == len(ws)
        batch_size:
            批量的大小
        raw:
            是否返回原始对象，如果为True，假设结果ret，那么len(ret) == len(data) * 3
            如果为False，那么len(ret) == len(data) * 2

    例如需要输入问题与答案的队列，问题队列Q = (q_1, q_2, q_3 ... q_n)
    答案队列A = (a_1, a_2, a_3 ... a_n)，有len(Q) == len(A)
    ws是一个Q与A共用的WordSequence对象，
    那么可以有： batch_flow([Q, A], ws, batch_size=32)
    这样会返回一个generator，每次next(generator)会返回一个包含4个对象的数组，分别代表：
    next(generator) == q_i_encoded, q_i_len, a_i_encoded, a_i_len
    如果设置raw = True，则：
    next(generator) == q_i_encoded, q_i_len, q_i, a_i_encoded, a_i_len, a_i

    其中 q_i_encoded 相当于 ws.transform(q_i)

    不过经过了batch修正，把一个batch中每个结果的长度，padding到了数组内最大的句子长度
    """

    all_data = list(zip(*data))

    if isinstance(ws, (list, tuple)):
        assert len(ws) == len(data), \
            'len(ws) must equal to len(data) if ws is list or tuple'

    if isinstance(add_end, bool):
        add_end = [add_end] * len(data)
    else:
        assert(isinstance(add_end, (list, tuple))), \
            'add_end 不是 boolean，就应该是一个list(tuple) of boolean'
        assert len(add_end) == len(data), \
            '如果 add_end 是list(tuple)，那么 add_end 的长度应该和输入数据长度一致'

    mul = 2
    if raw:
        mul = 3

    while True:
        data_batch = random.sample(all_data, batch_size)
        batches = [[] for i in range(len(data) * mul)]

        max_lens = []
        for j in range(len(data)):
            max_len = max([
                len(x[j]) if hasattr(x[j], '__len__') else 0
                for x in data_batch
            ]) + (1 if add_end[j] else 0)
            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)):
                if isinstance(ws, (list, tuple)):
                    w = ws[j]
                else:
                    w = ws

                # 添加结尾
                line = d[j]
                if add_end[j] and isinstance(line, (tuple, list)):
                    line = list(line) + [WordSequence.END_TAG]

                if w is not None:
                    x, xl = transform_sentence(line, w, max_lens[j], add_end[j])
                    batches[j * mul].append(x)
                    batches[j * mul + 1].append(xl)
                else:
                    batches[j * mul].append(line)
                    batches[j * mul + 1].append(line)
                if raw:
                    batches[j * mul + 2].append(line)
        batches = [np.asarray(x) for x in batches]

        yield batches



def batch_flow_bucket(data, ws, batch_size, raw=False,
                      add_end=True,
                      n_buckets=5, bucket_ind=1,
                      debug=False):
    """batch_flow的bucket版本
    多了两重要参数，一个是n_buckets，一个是bucket_ind
    n_buckets是分成几个buckets，理论上n_buckets == 1时就相当于没有进行buckets操作
    bucket_ind是指定哪一维度的输入数据作为bucket的依据
    """

    all_data = list(zip(*data))
    # for x in all_data:
    #     print(x[0][bucket_ind])
    #
    # lengths = 0
    lengths = sorted(list(set([len(x[0][bucket_ind]) for x in all_data])))
    if n_buckets > len(lengths):
        n_buckets = len(lengths)

    splits = np.array(lengths)[
        (np.linspace(0, 1, 5, endpoint=False) * len(lengths)).astype(int)
    ].tolist()
    splits += [np.inf]

    if debug:
        print(splits)

    ind_data = {}
    for x in all_data:
        l = len(x[0][bucket_ind])
        for ind, s in enumerate(splits[:-1]):
            if l >= s and l <= splits[ind + 1]:
                if ind not in ind_data:
                    ind_data[ind] = []
                ind_data[ind].append(x)
                break


    inds = sorted(list(ind_data.keys()))
    ind_p = [len(ind_data[x]) / len(all_data) for x in inds]
    if debug:
        print(np.sum(ind_p), ind_p)

    if isinstance(ws, (list, tuple)):
        assert len(ws) == len(data), \
            'len(ws) must equal to len(data) if ws is list or tuple'



    if isinstance(add_end, bool):
        add_end = [add_end] * len(data)
    else:
        assert(isinstance(add_end, (list, tuple))), \
            'add_end 不是 boolean，就应该是一个list(tuple) of boolean'
        assert len(add_end) == len(data), \
            '如果 add_end 是list(tuple)，那么 add_end 的长度应该和输入数据长度一致'

    mul = 2
    if raw:
        mul = 3

    while True:
        choice_ind = np.random.choice(inds, p=ind_p)
        if debug:
            print('choice_ind', choice_ind)
        data_batch = random.sample(ind_data[choice_ind], batch_size)
        batches = [[] for i in range(len(data) * mul)]

        max_lens = []
        for j in range(len(data)):
            max_len = max([
                len(x[j]) if hasattr(x[j], '__len__') else 0
                for x in data_batch
            ]) + (1 if add_end[j] else 0)
            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)):
                if isinstance(ws, (list, tuple)):
                    w = ws[j]
                else:
                    w = ws

                # 添加结尾
                line = d[j]
                if add_end[j] and isinstance(line, (tuple, list)):
                    line = list(line) + [WordSequence.END_TAG]

                if w is not None:
                    x, xl = transform_sentence(line, w, max_lens[j], add_end[j])
                    batches[j * mul].append(x)
                    batches[j * mul + 1].append(xl)
                else:
                    batches[j * mul].append(line)
                    batches[j * mul + 1].append(line)
                if raw:
                    batches[j * mul + 2].append(line)
        batches = [np.asarray(x) for x in batches]

        yield batches



# def test_batch_flow():
#     """test batch_flow function"""
#     from fake_data import generate
#     x_data, y_data, ws_input, ws_target = generate(size=10000)
#     flow = batch_flow([x_data, y_data], [ws_input, ws_target], 4)
#     x, xl, y, yl = next(flow)
#     print(x.shape, y.shape, xl.shape, yl.shape)
#
#
# def test_batch_flow_bucket():
#     """test batch_flow function"""
#     from fake_data import generate
#     x_data, y_data, ws_input, ws_target = generate(size=10000)
#     flow = batch_flow_bucket(
#         [x_data, y_data], [ws_input, ws_target], 4,
#         debug=True)
#     for _ in range(10):
#         x, xl, y, yl = next(flow)
#         print(x.shape, y.shape, xl.shape, yl.shape)
#
#
# if __name__ == '__main__':
#     test_batch_flow_bucket()

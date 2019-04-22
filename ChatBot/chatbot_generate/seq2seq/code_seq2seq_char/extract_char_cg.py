"""把 chicken_and_gossip数据 文件格式转换为可训练格式
Code from: QHDuan(2018-02-05) url: https://github.com/qhduan/just_another_seq2seq
"""

from conf.path_config import chicken_and_gossip_path
from conf.path_config import chatbot_data_cg_char_dir
from conf.path_config import chatbot_data_cg_ws_anti
from conf.path_config import chatbot_data_cg_xy_anti
from conf.path_config import model_ckpt_cg_anti

from utils.mode_util.seq2seq.word_sequence import WordSequence
from utils.text_tools import txtRead
from tqdm import tqdm
import pickle
import sys
import re

sys.path.append('..')


def make_split(line):
    """构造合并两个句子之间的符号
    """
    if re.match(r'.*([，。…？！～\.,!?])$', ''.join(line)):
        return []
    return ['，']


def good_line(line):
    if len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) > 2:
        return False
    return True


def regular(sen, limit=50):
    sen = re.sub(r'\.{3,100}', '…', sen)
    sen = re.sub(r'…{2,100}', '…', sen)
    sen = re.sub(r'[,]{1,100}', '，', sen)
    sen = re.sub(r'[\.]{1,100}', '。', sen)
    sen = re.sub(r'[\?]{1,100}', '？', sen)
    sen = re.sub(r'[!]{1,100}', '！', sen)
    if len(sen) > limit:
        sen = sen[0:limit]
    return sen


def creat_train_data_of_cg_corpus(limit=50, x_limit=2, y_limit=2):
    x_datas = []
    y_datas = []
    max_len = 0
    sim_ali_web_gov_dli_datas = txtRead(chicken_and_gossip_path, encodeType="utf-8")
    for sim_ali_web_gov_dli_datas_one in sim_ali_web_gov_dli_datas[1:]:
        if sim_ali_web_gov_dli_datas_one:
            sim_ali_web_gov_dli_datas_one_split = sim_ali_web_gov_dli_datas_one.strip().split("\t")
            if len(sim_ali_web_gov_dli_datas_one_split) == 2:
                # if sim_ali_web_gov_dli_datas_one_split[2]=="1":
                len_x1 = len(sim_ali_web_gov_dli_datas_one_split[0])
                len_x2 = len(sim_ali_web_gov_dli_datas_one_split[1])
                # if max_len < len_x1 or max_len < len_x2:
                max_len = max(len_x1, len_x2, max_len)

                sentence_org = regular(sim_ali_web_gov_dli_datas_one_split[0], limit=limit)
                sentence_sim = regular(sim_ali_web_gov_dli_datas_one_split[1], limit=limit)
                x_datas.append([sen for sen in sentence_org])
                y_datas.append([sen for sen in sentence_sim])
                # x_datas.append([sen for sen in sentence_sim])
                # y_datas.append([sen for sen in sentence_org])

    datas = list(zip(x_datas, y_datas))
    datas = [
        (x, y)
        for x, y in datas
        if len(x) < limit and len(y) < limit and len(y) >= y_limit and len(x) >= x_limit
    ]
    x_datas, y_datas = zip(*datas)

    print('fit word_sequence')

    ws_input = WordSequence()
    ws_input.fit(x_datas + y_datas)

    print('dump')

    pickle.dump((x_datas, y_datas),
                open(chatbot_data_cg_xy_anti, 'wb')
                )
    pickle.dump(ws_input, open(chatbot_data_cg_ws_anti, 'wb'))

    print('done')
    print(max_len)


if __name__ == '__main__':
    creat_train_data_of_cg_corpus()

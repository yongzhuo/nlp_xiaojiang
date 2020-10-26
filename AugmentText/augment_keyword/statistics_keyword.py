# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/10/25 11:07
# @author  : Mo
# @function: rule-word-freq, 统计各类别独有词汇的词频等


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
print(path_root)
# macadam
from utils.text_tools import jieba_cut, txtRead, txtWrite, load_json, save_json
from conf.path_config import stop_words_path
from collections import Counter, OrderedDict
from tqdm import tqdm
import jieba
import json
import copy


# 停用词列表，默认使用hanlp停用词表
f_stop = open(stop_words_path, "r", encoding="utf-8")
stop_words = []
for stop_word in f_stop.readlines():
    stop_words.append(stop_word.strip())

# stop_words = ["\t"]


def is_total_number(text: str) -> bool:
    """
    judge is total chinese or not, 判断是不是全是数字
    Args:
        text: str, eg. "macadam, 碎石路"
    Returns:
        bool, True or False
    """
    for word in text:
        if word not in "0123456789.%":
            return False
    return True


def statistics_keyword_by_label(path, rate=1):
    """
    judge is total chinese or not, 判断是不是全是数字
    Args:
        path: str, eg. "train.json"
        rate: float, eg. 0.75
    Returns:
        None
    """
    datas = txtRead(path)

    lwd = {}
    for i in tqdm(range(len(datas)), desc="jieba cut and statistics: "):
        # 从标准文档里边获取文本, 切词处理
        d = datas[i]
        d_json = json.loads(d)
        text = d_json.get("x", {}).get("text")
        label = d_json.get("y")
        word_list = list(jieba.cut(text))
        # 去除 停用词、全数字、1个字
        word_list = [wl for wl in word_list if wl not in stop_words and not is_total_number(wl) and len(wl) >= 2]
        # 词频统计(类别内)
        word_freq_dict = dict(Counter(word_list))
        if label not in lwd:
            lwd[label] = word_freq_dict
        else:
            lwd[label].update(word_freq_dict)

    # 取范围, 排序
    lwd_keys = list(lwd.keys())
    lwd_soft = [sorted(lwd[l].items(), key=lambda x: x[1], reverse=True) for l in lwd_keys]
    lwd_soft_rate = [s[:int(len(s) * rate)] for s in lwd_soft]
    label_word_dict = {lwd_keys[i]: OrderedDict(lwd_soft_rate[i]) for i in range(len(lwd_keys))}
    print("cut ok!")
    # 获取每个类独有的词汇
    label_keys = set(list(label_word_dict.keys()))
    label_words = {}
    for key in label_keys:
        key_dict = set(list(label_word_dict[key].keys()))
        keys_other = copy.deepcopy(label_keys)
        keys_other.discard(key)
        # 其他类别的所有词汇
        kos = set()
        for ko in keys_other:
            ko_dict = set(list(label_word_dict[ko].keys()))
            kos = kos | ko_dict

        # 获取独有的词汇
        key_public = kos & key_dict
        key_label = key_dict - key_public

        label_word_freq = {kl:label_word_dict[key][kl] for kl in key_label}
        label_words[key] = label_word_freq

    save_json(label_words, "label_keyword_unique.json")


if __name__ == '__main__':
    path = "ccks_news_2020.json"
    statistics_keyword_by_label(path, rate=1)
    mm = 0


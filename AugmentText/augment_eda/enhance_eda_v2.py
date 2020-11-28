# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/15 14:54
# @author  : Mo
# @function: EDA


# import macropodus
import synonyms
import random
import jieba


KEY_WORDS = ["macropodus"] # 不替换同义词的词语
ENGLISH = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def is_english(text):
    """
        是否全是英文
    :param text: str, like "你是谁"
    :return: boolean, True or False
    """
    try:
        text_r = text.replace(" ", "").strip()
        for tr in text_r:
            if tr in ENGLISH:
                continue
            else:
                return False
    except Exception as e:
        return False


def is_number(text):
    """
        判断一个是否全是阿拉伯数字
    :param text: str, like "1001"
    :return: boolean, True or False 
    """
    try:
        text_r = text.replace(" ", "").strip()
        for tr in text_r:
            if tr.isdigit():
                continue
            else:
                return False
    except Exception as e:
        return False


def get_syn_word(word):
    """
        获取同义词
    :param word: str, like "学生"
    :return: str, like "学生仔"
    """
    if not is_number(word.strip()) or not is_english(word.strip()):
        word_syn = synonyms.nearby(word)
        word_syn = word_syn[0] if len(word_syn[0]) else [word]
        return word_syn
    else:
        return [word]


def syn_replace(words, n=1):
    """
        同义词替换
    :param words: list, like ["macropodus", "是", "谁"]
    :param n: int, like 128
    :return: list, like ["macropodus", "是不是", "哪个"]
    """
    words_set = list(set(words)) # 乱序, 选择
    random.shuffle(words_set)
    count = 0
    for ws in words_set:
        if ws in KEY_WORDS or is_english(ws) or is_number(ws):
            continue  # 关键词/英文/阿拉伯数字不替换
        need_words = get_syn_word(ws) # 获取同义词(可能有多个)
        if need_words: # 如果有同义词则替换
            need_words = random.choice(need_words)
            words = [need_words if w==ws else w for w in words]
            count += 1
        if count >= n:
            break
    return words


def syn_insert(words, n=1, use_syn=True):
    """
        同义词替换
    :param words: list, like ["macropodus", "是", "谁"]
    :param n: int, like 32
    :return: list, like ["macropodus", "是不是", "哪个"]
    """
    words_set = list(set(words))  # 乱序, 选择
    random.shuffle(words_set)
    count = 0
    for ws in words_set:
        if ws in KEY_WORDS or is_english(ws) or is_number(ws):
            continue  # 关键词/英文/阿拉伯数字不替换
        if use_syn:
            need_words = get_syn_word(ws)  # 获取同义词(可能有多个)
        else:
            need_words = [ws]
        if need_words:  # 如果有同义词则替换
            random_idx = random.randint(0, len(words) - 1)
            words.insert(random_idx, (need_words[0]))
            count += 1
        if count >= n:
            break
    return words


def word_swap(words, n=1):
    """
        随机交换，随机交换两个词语
    :param words: list, like ["macropodus", "是", "谁"]
    :param n: int, like 2
    :return: list, like ["macropodus", "谁", "是"]
    """
    idxs = [i for i in range(len(words))]
    count = 0
    while count < n:
        idx_select = random.sample(idxs, 2)
        temp = words[idx_select[0]]
        words[idx_select[0]] = words[idx_select[1]]
        words[idx_select[1]] = temp
        count += 1
    return words


def word_delete(words, n=1):
    """
        随机删除N个词语
    :param words: list, like ["macropodus", "是", "谁"]
    :param n: int, like 1
    :return: list, like ["macropodus", "谁"]
    """
    count = 0
    while count < n:
        word_choice = random.choice(words)
        if word_choice not in KEY_WORDS:
            words.remove(word_choice)
            count += 1
    return words


def word_cut(text, tool="macropodus"):
    """
        切词工具
    :param text:str, like "macropodus是谁" 
    :param tool: str, "macropodus" or "jieba"
    :return: list, like ["macropodus", "是", "谁"]
    """
    if tool=="macropodus":
        text_cut = list(macropodus.cut(text))
    elif tool=="jieba":
        text_cut = list(jieba.cut(text))
    else:
        text_cut = list(jieba.cut(text))
    return text_cut


def eda(text, n=1, use_syn=True):
    """
        EDA, 每种方法进一位
    :param text: str, like "macropodus是谁" 
    :param n: int, like 1
    :param use_syn: Boolean, True or False
    :return: list, like ["macropodus是谁呀", "macropodus是"]
    """
    sens = word_cut(text, tool="jieba")
    # print(sens)
    sr = syn_replace(sens.copy(), n=n)
    si = syn_insert(sens.copy(), n=n, use_syn=use_syn)
    ws = word_swap(sens.copy(), n=n)
    wd = word_delete(sens.copy(), n=n)
    sens_word_4 = [sr, si, ws, wd]
    # print(sens_word_4)
    sens_4 = ["".join(s4) for s4 in sens_word_4]
    return sens_4


if __name__ == "__main__":
    sens = "".join(["macropodus", "是不是", "哪个", "啦啦",
                    "只需做好这四点，就能让你养的天竺葵全年花开不断！"])
    print(eda(sens))


    sens = list(sens)
    res1 = syn_replace(sens, n=1)
    print(res1)
    res2 = syn_insert(sens.copy(), n=1, use_syn=True)
    print(res2)
    res3 = word_swap(sens.copy(), n=1)
    print(res3)
    res4 = word_delete(sens.copy(), n=1)
    print(res4)



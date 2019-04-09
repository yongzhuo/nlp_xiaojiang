# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/1 10:35
# @author   :Mo
# @function :enhance text by eda, eda is replace, insert, swap, delete


from utils.text_tools import is_total_english
from utils.text_tools import is_total_number
from conf.path_config import stop_words_path
from utils.text_tools import jieba_cut
from random import shuffle
import synonyms
import random


random.seed(2019)
key_word_list = ["rsh", "mo", "大漠帝国"]


# 停用词列表，默认使用hanlp停用词表
f_stop = open(stop_words_path, "r", encoding="utf-8")
stop_words = []
for stop_word in f_stop.readlines():
    stop_words.append(stop_word.strip())


def synonym_replacement(words, n, key_words):
    """
      同义词替换,替换一个语句中的n个单词为其同义词
    :param words: list, inupt sentence
    :param n: int, replace words
    :return: list, new_words
    """
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        sim_synonyms = get_syn_by_synonyms(random_word)
        if len(sim_synonyms) >= 1 and random_word not in key_words and not is_total_english(random_word) and not is_total_number(random_word):
            synonym = random.choice(sim_synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    return new_words


def get_syn_by_synonyms(word):
    if not is_total_english(word.strip()):
        return synonyms.nearby(word)[0]
    else:
        return word


def random_insertion(words, n, key_words):
    """
      随机插入, 随机在语句中插入n个词
    :param words: list, inupt sentence
    :param n: int, insert words
    :return: list, new_words
    """
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words, key_words)
    return new_words


def add_word(new_words, key_words):
    """
      在list上随机插入一个同义词
    :param words: list, inupt sentence
    :return: list, new_words
    """
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        # 过滤
        if random_word not in key_words and not is_total_english(random_word) and not is_total_number(random_word):
            synonyms = get_syn_by_synonyms(random_word)
            counter += 1
        if counter >= 10:
            return
    random_synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


def random_swap(words, n):
    """
      随机交换，随机交换两个词语n次数
    :param words: list, inupt sentence
    :param n: int, swap words
    :return: list, new_words 
    """
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    """
        随机交换，随机交换两个词语
    :param new_words: list, inupt sentence
    :return: list, new_words 
    """
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_deletion(words, p, key_words):
    """
      随机删除,以概率p删除语句中的词
    :param words: list, inupt sentence
    :param p: float,随机删除的概率
    :return: list, 返回等
    """
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p or word in key_words:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


def sentence_replace_whitespace(sentences):
    """
      去除空格
    :param sentences: list,
    :return: list
    """
    sentences_new = []
    for sentence in sentences:
        sentence_replace = sentence.replace(" ", "").strip()
        sentences_new.append(sentence_replace + "\n")
    return sentences_new


def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9, key_words=[]):
    """
      EDA函数，同义词替换、插入词汇、交换词语顺序、删除词语
    :param sentence: str, input sentence
    :param alpha_sr: float, synonym_replacement
    :param alpha_ri: float, random_insertion
    :param alpha_rs: float, random_swap
    :param p_rd:     float, random_deletion
    :param num_aug:  int, generate n new sentence
    :return: list, contain orl sentence
    """
    seg_list = jieba_cut(sentence)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug*2 / 4) + 1
    n_sr = max(1, int(alpha_sr * num_words)) * 2
    n_ri = max(1, int(alpha_ri * num_words)) * 2
    n_rs = max(1, int(alpha_rs * num_words))

    # 同义词替换sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr, key_words)
        augmented_sentences.append(''.join(a_words))

    # 随机插入ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri, key_words)
        augmented_sentences.append(''.join(a_words))

    # 随机交换rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(''.join(a_words))

    # 随机删除rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd, key_words)
        augmented_sentences.append(''.join(a_words))

    augmented_sentences = list(set(augmented_sentences))
    shuffle(augmented_sentences)
    # 太短的句子不要
    augmented_sentences_new = []
    for augmented_sentences_one in augmented_sentences:
        if len(augmented_sentences_one) > 5:
            augmented_sentences_new.append(augmented_sentences_one)

    augmented_sentences = augmented_sentences_new
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    if len(augmented_sentences) > num_aug:
        augmented_sentences = augmented_sentences[0:num_aug]
    # augmented_sentences.append(seg_list)
    return augmented_sentences




if __name__ == "__main__":
    des = get_syn_by_synonyms("同义词")
    print(des)
    syn = eda(sentence="rsh喜欢大漠帝国吗", alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.2, p_rd=0.2, num_aug=10, key_words=key_word_list)
    syn_s = sentence_replace_whitespace(syn)
    print(syn)
    while True:
        print('输入: ')
        sen = input()
        syn = eda(sentence=sen)
        print(syn)
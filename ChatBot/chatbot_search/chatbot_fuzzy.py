# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/4 10:00
# @author   :Mo
# @function :


from conf.path_config import chicken_and_gossip_path
from utils.text_tools import txtRead, txtWrite
from conf.path_config import projectdir
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import pickle
import time
import re


def count_same_char(x1, x2):
    '''获取相同字符的个数'''
    res = []
    for x in x1:
      if x in x2:
        res.append(x)
    if res:
        return len(res)
    else:
        return 0


def fuzzy_re(user_input, collection):
    '''匹配方法， 效果不大好，只能匹配相同字数一样，或者字数比他多的那种，同义词或者是有一个词不一样，就没法区分开'''
    suggestions = []
    user_input = user_input.replace('.', '').replace('*', '').replace('?', '')

    collection_new = []
    len_user_input = len(user_input)
    for coll in collection:  # 获取包含所有字符的，如果不包含，就返回错误
        count_coll = 0
        for i in range(len_user_input):
            if user_input[i] in coll:
                count_coll += 1
        if len_user_input == count_coll:
            collection_new.append(coll)
    if not collection_new:
        return None


    pattern = '.*?'.join(user_input)  # Converts 'djm' to 'd.*?j.*?m'
    try:
        regex = re.compile(pattern)  # Compiles a regex.
    except:
        gg  = 0
    for item in collection_new:
        match = regex.search(item)  # Checks if the current item matches the regex.
        if match:
            suggestions.append((len(match.group()), match.start(), item))
    return [x for _, _, x in sorted(suggestions)]


def fuzzy_fuzzywuzzy(fuzz, user_input, collection):
    '''编辑距离，速度比较慢，比起匹配方法，能够处理字符不一样的问题'''
    collection_new = []
    len_user_input = len(user_input)
    for coll in collection:  # 获取包含一个字符的，如果不包含，就返回错误
        for i in range(len_user_input):
            if user_input[i] in coll:
                collection_new.append(coll)
    if not collection_new:
        return None
    collection_new = list(set(collection_new))

    same_char_list = []
    for collection_new_one in collection_new: # 获取相同字符串多的问题
        count_same_char_one = count_same_char(user_input, collection_new_one)
        same_char_list.append((collection_new_one, count_same_char_one))
    same_char_list.sort(key=lambda x: x[1], reverse=True)
    if len(same_char_list) >= 500:
        same_char_list = same_char_list[0: 500]

    result =  process.extract(user_input, same_char_list, scorer=fuzz.token_set_ratio, limit=20)
    return result


def fuzzy_fuzzywuzzy_list(fuzz, user_input, qa_list, collection, topn=50):
    '''编辑距离，速度比较慢，比起匹配方法，能够处理字符不一样的问题'''

    start_time = time.time()
    # user_input_set = set([user_input_one for user_input_one in user_input])
    user_input_set = [user_input_one for user_input_one in user_input]


    same_char_list = []
    max_data = 0
    max_data_list = []
    count_collection_new_one = 0
    for collection_new_one in collection: # 获取相同字符串多的问题
        count_same_char_one = len([x for x in user_input_set if x in collection_new_one])

        if count_same_char_one > 0:
            same_char_list.append((count_collection_new_one, count_same_char_one))
        if count_same_char_one > max_data:
            max_data_list.append(count_same_char_one)
            max_data = count_same_char_one
        count_collection_new_one += 1

    end_time1 = time.time()
    list_max_count = []
    len_max_data_list = len(max_data_list)
    for x in range(len_max_data_list):  # 获取前20排名
        for k,l in same_char_list:
            if l == max_data_list[len_max_data_list -1 - x]:
                list_max_count.append(qa_list[k]) #问答重这里取出来
        if len(list_max_count) >= 5000:
            list_max_count = list_max_count[0:5000]
            break

    end_time2 = time.time()

    # end_time1: 0.34090662002563477
    # end_time2: 0.4080846309661865

    # end_time1: 0.06417036056518555
    # end_time2: 0.08422374725341797

    # same_char_list.sort(key=lambda x: x[1], reverse=True)
    # if len(same_char_list) >= 20:
    #     same_char_list = same_char_list[0: 20]

    result =  process.extract(user_input, list_max_count, scorer=fuzz.token_set_ratio, limit=topn)
    end_time3 = time.time()

    # print('end_time1: ' + str(end_time1 - start_time))
    # print('end_time2: ' + str(end_time2 - start_time))
    # print('end_time3: ' + str(end_time3 - start_time))

    return result
    # [fuzz.WRatio, fuzz.QRatio,
    #  fuzz.token_set_ratio, fuzz.token_sort_ratio,
    #  fuzz.partial_token_set_ratio, fuzz.partial_token_sort_ratio,
    #  fuzz.UWRatio, fuzz.UQRatio]


if __name__ == '__main__':
    start_time = time.time()
    qa_list = txtRead(chicken_and_gossip_path)
    questions = [qa.strip().split("\t")[0] for qa in qa_list]
    print("read questions ok!")
    sen = "你谁呀"
    # list_fuzzyfinder = fuzzyfinder(base_syn_one_split[1], qa_list)
    # list_fuzzyfinder = fuzzy_fuzzywuzzy(fuzz, base_syn_one_split[1], qa_list)
    print("你问: " + "你谁呀")
    list_fuzzyfinder = fuzzy_fuzzywuzzy_list(fuzz, sen, qa_list, questions, topn=5)
    print("小姜机器人： " + list_fuzzyfinder[0][0].split("\t")[1].strip())
    print("推荐结果: ")
    print(list_fuzzyfinder)

    while True:
        print("你问: ")
        ques = input()
        list_fuzzyfinder = fuzzy_fuzzywuzzy_list(fuzz, ques, qa_list, questions, topn=5)
        print("小姜机器人： " + list_fuzzyfinder[0][0].split("\t")[1].strip())
        print("推荐结果: ")
        print(list_fuzzyfinder)

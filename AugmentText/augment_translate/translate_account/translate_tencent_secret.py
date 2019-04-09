# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/9 23:05
# @author   :Mo
# @function :使用腾讯账户（翻译君），回译


from conf.augment_constant import language_short_tencent
from conf.augment_constant import app_secret_tentcnet
from conf.augment_constant import app_key_tencent
from urllib.parse import quote
import logging as logger
import requests
import hashlib
import random
import string
import time
import json


def md5_sign(text):
    """
       生成md5
    :param src: str, sentence
    :return: str, upper of string
    """
    md5_model = hashlib.md5(text.encode("utf8"))
    return md5_model.hexdigest().upper()


def get_params(text, from_l="zh", to_l="en"):
    """
        生成sign和params
    :param text:  str, input sentence
    :param from_: source language
    :param to_:   target language
    :return:      dict, params
    """
    # 请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效）  
    time_stamp = str(int(time.time()))
    # 请求随机字符串，用于保证签名不可预测  
    nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 10))
    params = {'app_id': app_key_tencent,
              'source': from_l,
              'target': to_l,
              'text': text,
              'time_stamp': time_stamp,
              'nonce_str': nonce_str
              }
    signs = ''
    # 要对key排序再拼接  
    for key in sorted(params):
        # 键值拼接过程value部分需要URL编码，URL编码算法用大写字母，例如%E8。quote默认大写。  
        signs += '{}={}&'.format(key, quote(params[key], safe='').replace("%20", "+"))
    # 将应用密钥以app_key为键名，拼接到字符串sign_before末尾  
    signs += 'app_key={}'.format(app_secret_tentcnet)
    # 对字符串sign_before进行MD5运算，得到接口请求签名  
    sign = md5_sign(signs)
    params['sign'] = sign
    return params


def any_to_any_translate_tencent(text, from_='zh', to_='en'):
    """
       调用搜狗翻译，从任意一种语言到另外一种语言，详情见常量LANGUAGE_SHORT_BAIDU
    :param text:  str, input sentence
    :param from_: source language
    :param to_:   target language
    :return:      str, translate sentence
    """
    try:
        url = "https://api.ai.qq.com/fcgi-bin/nlp/nlp_texttranslate"
        params_text = get_params(text, from_l=from_, to_l=to_)
        res_post = requests.request("POST", url, data=params_text)
        res_content = res_post.content.decode("utf8")
        res_json = json.loads(res_content)
        target_text = res_json["data"]["target_text"]
        return target_text
    except Exception as e:
        logger.error(str(e))
        return None


def translate_tencent_back(text, from_='zh', to_='en'):
    """
       回译，调用两次腾讯翻译
    :param text:  str, input sentence
    :param from_: source language
    :param to_:   target language
    :return:      str, translate sentence
    """
    try:
        text_from_to = any_to_any_translate_tencent(text, from_=from_, to_=to_)
        text_to_from = any_to_any_translate_tencent(text_from_to, from_=to_, to_=from_)
        return text_to_from
    except Exception as e:
        logger.error(str(e))
        return None



if __name__ == '__main__':
    text_test = "你觉得JY会喜欢暗影随风、大漠帝国吗".strip()
    for to_test in language_short_tencent:
        res_test = translate_tencent_back(text_test, from_='zh', to_=to_test)
        print("没有账户就为空，回译结果: ")
        print(res_test)

# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/9/17 21:28
# @author  : Mo
# @function:


from __future__ import print_function, division, absolute_import, division, print_function

# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./."))
sys.path.append(path_root)
from argparse import Namespace
import requests
import json


from TFServing_preprocess import covert_text_to_id
from TFServing_postprocess import postprocess


def qa_tfserving(data_input, url):
    """  tf-serving 一整套流程 """
    bert_input = covert_text_to_id(data_input)
    data = json.dumps(bert_input)
    r = requests.post(url, data)
    r_text_json = json.loads(r.text)
    r_post = postprocess(r_text_json)
    return r_post


if __name__ == '__main__':
    data_input = {"data": [{"text": "别逗小通了!可怜的"}]}
    url = "http://192.168.1.97:8532/v1/models/chatbot_tf:predict"
    res = qa_tfserving(data_input, url)
    print(res)


    import os, inspect
    current_path = inspect.getfile(inspect.currentframe())
    path_root = "/".join(current_path.split("/")[:-1])
    print(path_root)
    print(current_path)
    print(inspect.currentframe())







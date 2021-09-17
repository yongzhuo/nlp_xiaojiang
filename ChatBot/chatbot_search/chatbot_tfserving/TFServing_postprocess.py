# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/4/15 21:59
# @author  : Mo
# @function: postprocess of TFServing, 后处理

from __future__ import print_function, division, absolute_import, division, print_function

# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./."))
sys.path.append(path_root)
from argparse import Namespace
import json


def load_json(path):
    """
      获取json，只取第一行
    :param path: str
    :return: json
    """
    with open(path, 'r', encoding='utf-8') as fj:
        model_json = json.load(fj)
    return model_json


#  字典
from bertWhiteConf import bert_white_config
config = Namespace(**bert_white_config)
id2answer = load_json(os.path.join(config.save_dir, config.path_answers))
id2doc = load_json(os.path.join(config.save_dir,config.path_docs))


def postprocess(predictions):
    """ 后处理 """
    predicts = predictions.get("predictions", {})
    token_ids = []
    for p in predicts:
        doc_id = str(p.get("doc_id", ""))
        score = p.get("score", "")
        answer = id2answer.get(doc_id, "")
        doc = id2doc.get(doc_id, "")
        token_ids.append({"score": round(score, 6), "doc": doc, "answer": answer, "doc_id": doc_id})
    return {"instances": token_ids}


if __name__ == '__main__':
    predictions = {"predictions": [
        {
            "score": 0.922845,
            "doc_id": 86
        },
        {
            "score": 0.922845,
            "doc_id": 104
        },
        {
            "score": 0.891189814,
            "doc_id": 101
        }
    ]}


    res = postprocess(predictions)
    print(res)


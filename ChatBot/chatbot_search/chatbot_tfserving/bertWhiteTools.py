# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/5/13 21:24
# @author  : Mo
# @function:


from typing import List, Dict, Union, Any
import logging as logger
import json


def txt_read(path: str, encoding: str = "utf-8") -> List[str]:
    """
    Read Line of list form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        dict of word2vec, eg. {"macadam":[...]}
    """

    lines = []
    try:
        file = open(path, "r", encoding=encoding)
        while True:
            line = file.readline().strip()
            if not line:
                break
            lines.append(line)
        file.close()
    except Exception as e:
        logger.info(str(e))
    finally:
        return lines


def txt_write(lines: List[str], path: str, model: str = "w", encoding: str = "utf-8"):
    """
    Write Line of list to file
    Args:
        lines: lines of list<str> which need save
        path: path of save file, such as "txt"
        model: type of write, such as "w", "a+"
        encoding: type of encoding, such as "utf-8", "gbk"
    """
    try:
        file = open(path, model, encoding=encoding)
        file.writelines(lines)
        file.close()
    except Exception as e:
        logger.info(str(e))


def save_json(jsons, json_path, indent=4):
    """
      保存json，
    :param json_: json
    :param path: str
    :return: None
    """
    with open(json_path, 'w', encoding='utf-8') as fj:
        fj.write(json.dumps(jsons, ensure_ascii=False, indent=indent))
    fj.close()


def load_json(path):
    """
      获取json，只取第一行
    :param path: str
    :return: json
    """
    with open(path, 'r', encoding='utf-8') as fj:
        model_json = json.load(fj)
    return model_json


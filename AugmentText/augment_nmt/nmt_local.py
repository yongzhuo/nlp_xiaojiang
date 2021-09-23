# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/9/22 21:37
# @author  : Mo
# @function: NMT of Helsinki-NLP
# 下载地址:
# opus-mt-en-zh:  https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
# opus-mt-zh-en:  https://huggingface.co/Helsinki-NLP/opus-mt-zh-en


from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, pipeline)
import time
import os


class BackTranslate:
    def __init__(self, pretrained_dir):
        # zh-to-en
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_dir, "Helsinki-NLP/opus-mt-zh-en"))
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(pretrained_dir, "Helsinki-NLP/opus-mt-zh-en"))
        # en-to-zh
        tokenizer_back_translate = AutoTokenizer.from_pretrained(os.path.join(pretrained_dir, "Helsinki-NLP/opus-mt-en-zh"))
        model_back_translate = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(pretrained_dir, "Helsinki-NLP/opus-mt-en-zh"))
        # pipeline
        self.zh2en = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer)
        self.en2zh = pipeline("translation_en_to_zh", model=model_back_translate, tokenizer=tokenizer_back_translate)

    def back_translate(self, text):
        """ 回译 """
        text_en = self.zh2en(text, max_length=510)[0]["translation_text"]
        print("text_en:", text_en)
        text_back = self.en2zh(text_en, max_length=510)[0]["translation_text"]
        print("text_back:", text_back)
        return text_back


if __name__ == '__main__':


    pretrained_dir = "D:/soft_install/dataset/bert-model/translate"
    bt = BackTranslate(pretrained_dir)
    datas = [{"text": "平乐县，古称昭州，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭城，总面积1919.34平方公里。"},
     {"text": "平乐县主要旅游景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐县为漓江分界点，平乐以北称漓江，以南称桂江，是著名的大桂林旅游区之一。"},
     {"text": "印岭玲珑，昭水晶莹，环绕我平中。青年的乐园，多士受陶熔。生活自觉自治，学习自发自动。五育并重，手脑并用。迎接新潮流，建设新平中"},
     {"text": "桂林山水甲天下, 阳朔山水甲桂林"},
     {"text": "三国一统天下"},
     {"text": "世间万物皆系于其上"},
     {"text": "2020年真是一个糟糕的年代, 进入20年代，新冠爆发、经济下行，什么的都来了。"},
     {"text": "仿佛一切都变得不那么重要了。"},
     {"text": "苹果多少钱一斤"}
    ]
    time_start = time.time()
    for da in datas:
        text = da.get("text", "")
        bt.back_translate(text)
    time_total = time.time() - time_start
    print("time_total:{}".format(time_total))
    print("time_per:{}".format(time_total / len(datas)))

    while True:
        print("请输入:")
        ques = input()
        res = bt.back_translate(ques)
        print("####################################################")


# 下载地址:
# opus-mt-en-zh:  https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
# opus-mt-zh-en:  https://huggingface.co/Helsinki-NLP/opus-mt-zh-en


# 备注: 翻译效果不大好



"""
text_en: Ping Lei County, anciently known as Zhao County, belongs to the city of Gui Lin, Guangxi Liang Autonomous Region, and is located in the north-east of Guangxi, south-east of the city of Gui Lin, eastern Pingshan County, south-west Su Ping, north-west of Yangyon and north-west of the city of Lilongqi, with a total area of 1919.34 square kilometres.
text_back: 平莱县,古代称为赵县,属于广西梁自治区Gui Lin市,位于广西东北、Gui Lin市东南、Pingshan县东南、Su Ping西南、Yangyon西北和Lilongqi市西北,总面积1919.34平方公里。
text_en: The main tourist attractions in the district of Ping Lei are Xin Xianjin Quan, Cold Water Qing Qing, Qingjiang, Qingjiang, Qingjiang, etc. The district of Ping Le is one of the well-known Grand Gui Lin tourist areas, which is known as Jingjiang, north of Ping Lei and south of Ping Lei.
text_back: 平莱区的主要旅游景点为新贤进泉、冷水清清、青江、青江、青江、青江等。 平来区是著名的大桂林旅游区之一,称为青江,位于平莱以北和平莱以南。
text_en: The young man's garden, the Doss, is molten with pottery. Life is self-governing, learning self-involvement. It's full and heavy, and the hands and brains work together. It takes a new tide and builds a new flat.
text_back: 年轻人的花园,多斯人,被陶器熔化了。生活是自治的,学习自我参与。生活是满的和沉重的,手和大脑一起工作。它需要新的潮水,建造新的公寓。
text_en: Guilin Mountain Watermarin, Sunshaw Hill Watermarin
text_back: 古林山水马林、桑肖山水马林
text_en: All three of us.
text_back: 我们三个人
text_en: Everything in the world is in it.
text_back: 世界上所有的东西都在里面
text_en: The year 2020 was a really bad time, and in the 20s, the crown broke out, the economy went down, everything came up.
text_back: 2020年是一个非常糟糕的时期, 在20年代,王冠崩盘, 经济下滑,一切都出现了。
text_en: As if everything had become less important.
text_back: 仿佛一切都变得不重要了
text_en: How much is an apple?
text_back: 苹果多少钱?
"""


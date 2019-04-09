# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/9 23:05
# @author   :Mo
# @function :使用翻译工具translate.Translator，回译


from conf.augment_constant import language_short_google
from utils.text_tools import judge_translate_english
from translate import Translator


def translate_tools_translate(text, to_='en'):
    """
       调用translate进行句子生成
    :param text: str, input
    :param to_: language type
    :return: str, result
    """
    # provider = 'mymemory','microsoft'
    translator1 = Translator(to_lang=to_, from_lang='zh', provider=None, secret_access_key=None)
    translator2 = Translator(to_lang="zh", from_lang=to_, provider=None, secret_access_key=None)

    translation1 = translator1.translate(text)
    translation2 = translator2.translate(translation1)
    return translation2


if __name__ == "__main__":
    sen_org = "大漠帝国喜欢RSH、JY吗"
    for language_short_google_one in language_short_google:
        text_translate = translate_tools_translate(sen_org, to_=language_short_google_one)
        judge = judge_translate_english(sen_org, text_translate)
        if judge:
            print("True")
            print(text_translate)
        else:
            print("False")
            print(text_translate)
# 测试结果:
# False
# 沙漠帝国是否像RSH，JY？
# False
# 沙漠帝国看起来像RSH，JY？
# False
# 帝国沙漠像rsh，jy？
# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/9 22:49
# @author   :Mo
# @function :get main_part by stanfordcorenlp


from conf.path_config import stanford_corenlp_full_path
from stanfordcorenlp import StanfordCoreNLP


# stanford-corenlp-full-2018-10-05需要预先下载，启动较慢
nlp = StanfordCoreNLP(stanford_corenlp_full_path, lang='zh')


def stanford_parse(sentence):
    tokenize = nlp.word_tokenize(sentence)
    pos_tag = nlp.pos_tag(sentence)
    name_entity = nlp.ner(sentence)
    syntax_tree = nlp.parse(sentence)
    dependence = nlp.dependency_parse(sentence)

    result_dict = {}
    result_dict['tokenize'] = tokenize
    result_dict['dependence'] = dependence
    result_dict['parse'] = syntax_tree
    return result_dict


def combine_nn(tokenize, dependence, target):
    """
      合并名词短语等
    :param dependence: dict, enhancedPlusPlusDependencies
    :param target: str, subject or object 
    :return: str, nn
    """
    if not target:
        return target
    else:
        for dependence_one in dependence:
            if target == tokenize[dependence_one[1]-1] if dependence_one[1]!=0 else "root" and dependence_one[0] == "nn":
                target = tokenize[dependence_one[2]-1] + target
                return target
    return target


def get_main_part_by_stanfordcorenlp(text):
    """
      根据依存句法生成句子
    :param text: str, 输入
    :return: str, result of syn sentence
    """
    # standcoreNLP 分词
    result_dict = stanford_parse(text)
    tokenize = result_dict['tokenize']
    dependence = result_dict['dependence']
    syntax_tree = result_dict['parse']
    # 提取主谓宾
    part_main = {"主": "", "谓": "", "宾": ""}
    if len(syntax_tree) >= 2:
        if "NP" in syntax_tree[1] or "ROOT" not in str(dependence):  # 名词短语 或者是没有谓语
            count = 0
            for syntax_tree_single in syntax_tree:
                if "NP" in syntax_tree_single and "(" in syntax_tree_single and ")" in syntax_tree_single:
                    token_np = syntax_tree_single.split(" ")[-1]
                    token_np = token_np.replace("'", "").replace(")", "").strip()
                    part_main["主"] = token_np if count == 0 else part_main["主"] + token_np
                    count += 1
            return part_main["主"] + part_main["谓"] + part_main["宾"]
        else:
            for dependence_one in dependence:
                dep = dependence_one[0]
                dep_dep_gloss = tokenize[dependence_one[2]-1]
                if dep == "ROOT":  # ROOT作谓语
                    part_main["谓"] = dep_dep_gloss
                elif dep == "cop":  # 主系结构
                    part_main["谓"] =  dep_dep_gloss + part_main["谓"]
                else:  # 主语和宾语
                    if dep == "nsubjpass" or dep == "dobj" or dep == "attr":
                        part_main["宾"] = dep_dep_gloss
                    elif dep == "nsubj" or dep == "top":
                        part_main["主"] = dep_dep_gloss

                part_main["主"] = combine_nn(tokenize, dependence, part_main["主"])
                part_main["宾"] = combine_nn(tokenize, dependence, part_main["宾"])
    return part_main["主"] + part_main["谓"] + part_main["宾"]


if __name__ == "__main__":
    sentence_list = [
        "大漠帝国确实很喜欢JY",
        "JY也喜欢大漠帝国哦！",
        "这个工程的作者是momo",
        "momo是一个无门无派的浪人",
        "只有自信的程序员才能把握未来",
        "主干识别可以提高检索系统的智能",
        "打更的住在这里",
        "人民的名义",
        "名词短语",
        "我一直很喜欢你",
        "你被我喜欢",
        "美丽又善良的你被卑微的我深深的喜欢着……",

        "搜索momo可以找到我的博客",
        "静安区体育局2013年部门决算情况说明",
        "这类算法在有限的一段时间内终止",
        "为什么在espace输入名字的部分拼音搜索不到相关的联系人？",
        "espace安装失败报错",
        "espace3.0无法批量导入",
        "espace拼音模糊搜索",
        "在运营数据中心内，ITSM系统的用户添加和更改了哪种类型的服务？",
        "在ITSM系统中申请服务请求时，流转过程中，如何在当前环节转至同组其他人处理工单？",
        "红旗飘",
        "柳丝长",
        "乐队奏国歌",
        "红扑扑的朝霞露出了笑脸",
        "初升的太阳照耀着峻峭的群山",
        "一个农人在路上看见一条冻僵了的蛇",
        "我打量了他一眼", ]
    sentence_type = ["陈述句与否定句",
            "秦耕真是一个聪明的孩子",
            "衣服洗得不干净",
            "他没有做完作业",
            "他不敢不来",
            "没有一个人不怕他",
            "我非把这本书读完不可",
            "同学们无不欢欣鼓舞",
            "他妈妈不让他去,无非是怕他吃亏",
            "想起一个人的旅途,不无寂寥之感",
            "你未必不知道",
            "各种问句",
            "你可以那到100分, 是吗?",
            "刚才接你的人是谁?",
            "什么叫函数?",
            "你爸爸怎么样了?",
            "你每天几点休息?",
            "你爸爸在哪儿?",
            "我们是从广州走, 还是从成都走?",
            "他是不是又迟到了?",
            "难道他已经跑了?",
            "我怎么能负这个责任呢?",
            "你是来帮助我们的, 还是来拆我们的台的?",
            "这些人甘愿当走狗, 你说可恨不可恨?",
            "祈使句",
             "快去捞饭!米烂了!",
             "给我喝水, 我渴!",
             "走哇, 妈妈!",
             "不许动!",
             "太好啦",
    ]
    for sen_one in sentence_list:
        subject_object = get_main_part_by_stanfordcorenlp(sen_one)
        print(sen_one + "   " + subject_object)

    while True:
        print("请输入sentence： ")
        sen_test = input()
        # syn_sentence_test = syn_by_syntactic_analys==(test_test)
        syn_sentence_test = get_main_part_by_stanfordcorenlp(sen_test)
        print(syn_sentence_test)

    # Do not forget to close! The backend server will consume a lot memery
    nlp.close()
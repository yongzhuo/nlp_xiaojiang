# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/4/3 11:23
# @author   :Mo
# @function :utils, tools


from openpyxl import Workbook
import logging as logger
import gensim
import jieba
import time
import xlrd
import re


#中英文标点符号
filters='[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' + '！，；：。？、“”’‘《》（）~@#￥%……&*\（）/{}【】…=-]'
#标点符号、空格
filters_1 = "[\.\!\/_,?;:$%^*<>()+\"\']+|[！，；：。？、“”’‘《》（）~@#￥%……&*\（\）\/\-]+"

"""去除标点符号、空格"""
def clear_punctuation(text):
    """去除标点符号"""
    sentence = text.replace(' ', '')
    sentence_punctuation_clear = re.sub(filters, ' ', sentence).strip()
    sentence_punctuation_clear_replace = sentence_punctuation_clear.replace('   ', ' ').replace('  ', ' ')
    return sentence_punctuation_clear_replace


'''截取中文、拼音、数字，去除特殊字符等'''
def getChinese1(ques):
    # ques = '•“鑫菁英”教育分期手续费怎么收取？可以'
    findAllChinese = ''.join(re.findall(u"([\u4e00-\u9fa50-9A-Za-z])", ques))
    # print(sub_str)
    return findAllChinese


'''xlrd读xls'''
def xlsRead(sheetName=None, cols=0, fileXlsPath=None):
    '''读xls文件'''
    workbook = xlrd.open_workbook(fileXlsPath)
    # 根据sheet索引或者名称获取sheet内容
    sheet = workbook.sheet_by_name(sheetName)
    nrows = sheet.nrows
    ncols = sheet.ncols

    listRows = []
    for i in range(nrows):
        listRows.append(sheet.row_values(i))

    return listRows


'''openpyxl写xlsx'''
def xlsxWrite(sheetName, writeList, fileXlsName):
    wb = Workbook()
    print('{}'.format(wb.get_sheet_names()))  # 提供一个默认名叫Sheet的表，office2016下新建提供默认Sheet1
    sheet = wb.create_sheet(sheetName)
    # i = 0
    for listLine_one in writeList:
        # i += 1
        sheet.append(listLine_one)
        # if i == 1000:
        #     break
    wb.save(fileXlsName)



"""判断一个unicode是否是英文字母"""
def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False

'''读取txt文件'''
def txtRead(filePath, encodeType = 'utf-8'):
    listLine = []
    try:
        file = open(filePath, 'r', encoding= encodeType)

        while True:
            line = file.readline()
            if not line:
                break

            listLine.append(line)

        file.close()

    except Exception as e:
        logger.info(str(e))

    finally:
        return listLine

'''读取txt文件'''
def txtWrite(listLine, filePath, type = 'w',encodeType='utf-8'):

    try:
        file = open(filePath, type, encoding=encodeType)
        file.writelines(listLine)
        file.close()

    except Exception as e:
        logger.info(str(e))

'''截取中文、拼音、数字，去除特殊字符等'''
'''要保留特殊字符的格式，最好的方法是每个字符都去匹配'''

def getChinese(ques):
    # ques = '•“鑫菁英”教育分期手续费怎么收取？可以'
    ques = strQ2B(ques)
    answer = ''
    for ques_one in ques:
        ques_one_findall = ''.join(re.findall(u"([\u4e00-\u9fa50-9A-Za-z峣㒶㒰玘宸諕鄕缓緩𪥵嬆嬲煙草砼赟贇龘㗊㵘㙓敠])", ques_one))
        if not ques_one_findall:
            ques_one_findall = ' '
        answer = answer + ques_one_findall
    answer = answer.strip().replace('  ', ' ').replace('   ', ' ')
    return answer.upper()

'''去除标点符号'''

def get_syboml(ques):
    # ques = '•“鑫菁英”教育分期手续费怎么收取？可以'
    ques = strQ2B(ques)
    # answer = re.sub(u'([。.,，、\；;：:？?！!“”"‘’'''（）()…——-《》<>{}_~【】\\[])', ' ', ques).replace('  ', ' ').replace('   ', ' ')
    answer = re.sub("[\.\!\/_,?;:$%^*<>()+\"\']+|[！，；：。？、“”’‘《》[\]（|）{}【】~@#￥%…&*\/\-—_]+", " ", ques).strip()
    return answer

'''xlrd读xls'''

def xlsRead(sheetName=None, cols=0, fileXlsPath=None):
    '''读xls文件'''
    workbook = xlrd.open_workbook(fileXlsPath)
    # 根据sheet索引或者名称获取sheet内容
    sheet = workbook.sheet_by_name(sheetName)
    nrows = sheet.nrows
    ncols = sheet.ncols

    listRows = []
    for i in range(nrows):
        listRows.append(sheet.row_values(i))

    return listRows

'''openpyxl写xlsx'''

def xlsxWrite(sheetName, writeList, fileXlsName):
    wb = Workbook()
    print('{}'.format(wb.get_sheet_names()))  # 提供一个默认名叫Sheet的表，office2016下新建提供默认Sheet1
    sheet = wb.create_sheet(sheetName)
    # i = 0
    for listLine_one in writeList:
        # i += 1
        sheet.append(listLine_one)
        # if i == 1000:
        #     break
    wb.save(fileXlsName)

'''读取txt文件'''

def txtRead(filePath, encodeType='utf-8'):
    listLine = []
    try:
        file = open(filePath, 'r', encoding=encodeType)

        while True:
            line = file.readline()
            if not line:
                break

            listLine.append(line)

        file.close()

    except Exception as e:
        logger.info(str(e))

    finally:
        return listLine

'''读取txt文件'''

def txtWrite(listLine, filePath, type='w', encodeType='utf-8'):

    try:
        file = open(filePath, type, encoding=encodeType)
        file.writelines(listLine)
        file.close()

    except Exception as e:
        logger.info(str(e))

# -*- coding: cp936 -*-
def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += chr(inside_code)
    return rstring

def is_valid_date(strdate):
    '''判断是否是一个有效的日期字符串'''
    try:
        if ":" in strdate:
            time.strptime(strdate, "%Y-%m-%d %H:%M:%S")
        else:
            time.strptime(strdate, "%Y-%m-%d")
        return True
    except:
        return False

'''判断是否是全英文的'''

def is_total_english(text):
    """判断一个是否是全英文字母"""
    symbol = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    try:
        sentence_punctuation_clear = get_syboml(text)
        sentence_punctuation_clear = sentence_punctuation_clear.replace(' ', '').strip()
        numben = 0
        for one in sentence_punctuation_clear:
            if one in symbol:
                numben += 1
        if numben == len(sentence_punctuation_clear):
            return True
        else:
            return False
    except:
        return False

'''判断是否是数字的'''

def is_total_number(text):
    """判断一个是否是全英文字母"""
    try:
        sentence_punctuation_clear = get_syboml(text)
        sentence_punctuation_clear = sentence_punctuation_clear.replace(' ', '').strip()
        numben = 0
        for one in sentence_punctuation_clear:
            if one.isdigit():
                numben += 1
        if numben == len(sentence_punctuation_clear):
            return True
        else:
            return False
    except:
        return False

def is_number_or_english(text):
    '''不为数字不为字母'''
    judge = False
    try:
        sentence_punctuation_clear = get_syboml(text)
        sentence_punctuation_clear = sentence_punctuation_clear.replace(' ', '').strip()
        for words in sentence_punctuation_clear:
            judge_number = is_total_number(words)
            judge_english = is_total_english(words)
            judge = judge_number or judge_english
            if not judge:
                return False
        return judge
    except:
        return False

def jieba_cut(text):
    """
      Jieba cut
    :param text: input sentence
    :return: list
    """
    return list(jieba.cut(text, cut_all=False, HMM=True))


def judge_translate_english(sen_org, sen_tra):
    """
      判断翻译后句子带英文的情况
    :param sen_org: str, 原始句子
    :param sen_tra: str, 翻译后的句子
    :return: boolean, True or False
    """
    # sen_org_cut = jieba_cut(sen_org)
    sen_tra_cut = jieba_cut(sen_tra)
    for sen_tra_cut_one in sen_tra_cut:
        if is_total_english(sen_tra_cut_one) and sen_tra_cut_one not in sen_org:
            return False
    return True


def load_word2vec_model(model_path, binary_type=True, encoding_type = 'utf-8', limit_words=None):
    '''
      下载词向量
    :param model_path: str
    :return:  word2vec model
    '''
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=binary_type, limit=limit_words, encoding=encoding_type, unicode_errors='ignore')
    return word2vec_model


#todo #句子改写，同义词替换，去停用词等


if __name__ == '__main__':


    # for i in range(10):
    #     sentence_vec = word2vec_model.wv["的"]
    #     sentence_vec_pd = pd.DataFrame(sentence_vec)
    #     sentence_vec_pd.to_csv('my_csv.csv', mode='a', header=False)

    # sentence_ee = pd.read_csv('my_csv.csv')

    # txtWrite([str(sentence_vec)], "gg.txt")


    # path_test_data_government = '/data/test_data_government.csv'
    # sentences = txtRead(path_test_data_government)
    sentences = []
    sentences_one_clear_punctuation_all = []
    for sentences_one in sentences[1:]:
        sentences_one_1 = sentences_one
        sentences_one_clear_punctuation = clear_punctuation(sentences_one_1.replace(',0.0,1.0', ''))
        # print(sentences_one)
        # print(sentences_one_clear_punctuation)
        sentences_one_clear_punctuation_jieba = jieba.cut(sentences_one_clear_punctuation, cut_all=False, HMM=False)
        sentences_one_clear_punctuation_jieba_list = ' '.join(list(sentences_one_clear_punctuation_jieba)).replace('   ', ' ').replace('  ', ' ').strip()
        sentences_one_clear_punctuation_all.append(sentences_one_clear_punctuation_jieba_list + ',0.0,1.0' + '\n')

    txtWrite(sentences[0:1] + sentences_one_clear_punctuation_all, '/data/test_data_government_cut.csv')

    #',0.0,1.0'
    # np.savetxt('001', [word2vec_model.wv["的"], word2vec_model.wv["的"]])
    # gg = np.loadtxt('001')

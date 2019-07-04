# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/13 10:17
# @author   :Mo
# @function :setup of nlp_xiaojiang
# @codes    :copy from https://github.com/TianWenQAQ/Kashgari/blob/master/setup.py

from setuptools import find_packages, setup
import pathlib

# Package meta-data.
NAME = 'nlp-xiaojiang'
DESCRIPTION = 'nlp of augment、chatbot、classification and featureproject of chinese text'
URL = 'https://github.com/yongzhuo/nlp_xiaojiang'
EMAIL = '1903865025@qq.com'
AUTHOR = 'yongzhuo'
LICENSE = 'MIT'

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

required = [
            'scikit-learn>=0.19.1',
            'fuzzywuzzy>=0.17.0',
            'openpyxl>=2.6.2',
            'xpinyin>=0.5.6',
            'gensim>=3.7.1',
            'jieba>=0.39',
            'xlrd>=1.2.0',
            'tensorflow>=1.8.0',
            'keras-bert>=0.41.0',
            'Keras>=2.2.0',
            'pandas>=0.23.0',
            'h5py>=2.7.1',
            'numpy>=1.16.1',
            'pyemd==0.5.1',
            'pathlib',
            'translate',
            'PyExecJS',
            'stanfordcorenlp',]

setup(name=NAME,
        version='0.0.1',
        description=DESCRIPTION,
        long_description=README,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        packages=find_packages(exclude=('tests')),
        install_requires=required,
        license=LICENSE,
        classifiers=['License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3.4',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: Implementation :: CPython',
                     'Programming Language :: Python :: Implementation :: PyPy'],)


if __name__ == "__main__":
    print("setup ok!")

# 说明，项目工程目录这里nlp_xiaojiang，实际上，下边还要有一层nlp_xiangjiang，也就是说，nlp_xiangjiang和setup同一层
# Data包里必须要有__init__.py，否则文件不会生成

# step:
#     打开cmd
#     到达安装目录
#     python setup.py build
#     python setup.py install
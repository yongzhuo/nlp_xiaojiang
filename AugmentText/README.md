# AugmentText

# 概述
    - 相较于图像数据增强，文本数据增强，现在还是有很多问题的；
    - 往更严格的角度看，文本数据增强更像是同义句生成，但又不完全是，它是一个更大范围的概念；
    - 很多时候，需要文本数据增强，一个是常常遇到的数据不足，另一个就是数据不均衡。
    - 我的感觉是，文本数据增强的有效方法:
    - 一个是回译（翻译两次，例如中文到英文，然后英文再到中文），
    - 另外一个就是EDA（同义词替换、插入、交换和删除），插入和交换当时确实没有想到用


###github项目地址为###
    https://github.com/yongzhuo/nlp_xiaojiang/tree/master/AugmentText


# 回译（相对靠谱）
    - 1.在线翻译工具（中文->[英、法、德、俄、西班牙、葡萄牙、日、韩、荷兰、阿拉伯]等语言）
       - 谷歌翻译(google)，谷歌翻译不用说，应该是挺好的，语言支持最多，不过我暂时还不会翻墙注册账户
       - 百度翻译(baidu)，百度翻译不用说，国内支持翻译语言最多的了(28种互译)，而且最大方了，注册账户后每月有200万字符的流量，大约是2M吧，超出则49元人民币/百万字符
       - 有道翻译(youdao)，初始接触网络的时候我最喜欢用有道翻译了，但死贵，只有100元体验金，差评。才支持11种语言，48元/百万字符
       - 搜狗翻译(sougou)，对于搜狗印象还行吧，毕竟是能做搜索引擎的公司嘛。78种语言，200元体验金，常见语言40元/百万字符,非常见语言60元/百万字符
       - 腾讯翻译(tencent)，总觉得腾讯AI是后知后觉了，公司调用腾讯接口老是变来变去的，这次也是被它的sign加密给恶心到了，空格改为+。或许对企鹅而言，人工智能不那么重要吧。
                          -有两个，一个是翻译君一个是AIlab什么的，支持的语言少些。似乎还在开发中，不限额不保证并发，php开发没有python的demo
       - 必应翻译(bing)，微软的东西，你懂的，没有尝试，直接在网页上试试还可以吧
       - 可以采用工具、模拟访问网页、或者是注册账号等
    - 2.离线翻译工具
       - 1.自己写，收集些语料，seq2seq,nmt,transformer
       - 2.小牛翻译，比较古老的版本了，win10或者linux都可以，不过只有训练好的中英互译
             地址:http://www.niutrans.com/index.html

# 同义词替换（还行）
    - 1.eda(其实就是同义词替换、插入、交换和删除)   论文《Easy data augmentation techniques for boosting performance on text classification tasks》
        - 中文实现的demo，github项目zhanlaoban/eda_nlp_for_Chinese，地址:https://github.com/zhanlaoban/eda_nlp_for_Chinese
    - 2.word2vec、词典同义词替换
        - 不同于1中使用synonyms工具查找同义词，可以使用gensim的词向量，找出某个词最相似的词作为同意词。
        - 还可以使用同义词典机械查找，词典可用fighting41love/funNLP，github地址:https://github.com/fighting41love/funNLP/tree/master/data/

# 句法、句子扩充、句子缩写（比较困难、）
    - 1.句子缩写，查找句子主谓宾等
        - 有个java的项目，调用斯坦福分词工具(不爱用)，查找主谓宾的
        - 地址为:（主谓宾提取器）https://github.com/hankcs/MainPartExtractor
    - 2.句子扩写  todo
    - 3.句法 todo

# HMM-marko（质量较差）
    - HMM生成句子原理: 根据语料构建状态转移矩阵，jieba等提取关键词开头，生成句子
        - 参考项目:https://github.com/takeToDreamLand/SentenceGenerate_byMarkov

# 深度学习方法 todo
    - seq2seq
    - bert
    - transformer
    - GAN
    
## 预训练模型-UMILM
  使用BERT(UNILM)的生成能力, 即BERT的NSP句对任务 
    - simbert(bert + unilm + adv):  [https://github.com/ZhuiyiTechnology/simbert](https://github.com/ZhuiyiTechnology/simbert)
    - simbert: [鱼与熊掌兼得：融合检索和生成的SimBERT模型](https://spaces.ac.cn/archives/7427)
    - roformer-sim:  [https://github.com/ZhuiyiTechnology/roformer-sim](https://github.com/ZhuiyiTechnology/roformer-sim)
    - simbert-v2(roformer + unilm + adv + bart + distill): [SimBERTv2来了！融合检索和生成的RoFormer-Sim模型](https://spaces.ac.cn/archives/8454)
    
## 回译(开源模型效果不是很好)
  中文转化成其他语言(如英语), 其他语言(如英语)转化成中文, Helsinki-NLP开源的预训练模型
    - opus-mt-en-zh:  https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
    - opus-mt-zh-en:  https://huggingface.co/Helsinki-NLP/opus-mt-zh-en


# 参考/感谢
* eda_chinese：[https://github.com/zhanlaoban/eda_nlp_for_Chinese](https://github.com/zhanlaoban/eda_nlp_for_Chinese)
* 主谓宾提取器：[https://github.com/hankcs/MainPartExtractor](https://github.com/hankcs/MainPartExtractor)
* HMM生成句子：[https://github.com/takeToDreamLand/SentenceGenerate_byMarkov](https://github.com/takeToDreamLand/SentenceGenerate_byMarkov)
* 同义词等：[https://github.com/fighting41love/funNLP/tree/master/data/](https://github.com/fighting41love/funNLP/tree/master/data/)
* 小牛翻译：[http://www.niutrans.com/index.html](http://www.niutrans.com/index.html)

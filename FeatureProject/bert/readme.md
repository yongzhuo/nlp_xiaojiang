# BERT

# usage:
this bert you need not fine tuning for common target

  * step1:github项目中只是上传部分数据，需要的前往链接: https://pan.baidu.com/s/1I3vydhmFEQ9nuPG2fDou8Q 提取码: rket
        download  chinese_L-12_H-768_A-12（谷歌预训练好的模型）
        解压到Data/chinese_L-12_H-768_A-12
  * step2-1:
       运行 FeatureProject/bert/extract_keras_bert_feature.py
       then you can get vector of bert encoding
  * step2-2:
       运行 FeatureProject/bert/tet_bert_keras_sim.py
       then you can get sim of bert vector of two sentence
                and get avg time of run a sentence of encode

# thanks
* keras-bert: https://github.com/CyberZHG/keras-bert

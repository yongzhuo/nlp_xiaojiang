# chatbot_bertwhite
## 解释说明
- 代码说明:
  - 1. bertWhiteConf.py  超参数配置, 地址、bert-white、索引工具等的超参数
  - 2. bertWhiteTools.py 小工具, 主要是一些文档读写功能函数
  - 3. bertWhiteTrain.py 主模块, 类似bert预训练模型编码
  - 4. indexAnnoy.py     annoy索引
  - 5. indexFaiss.py     faiss索引
  - 6. mmr.py            最大边界相关法, 保证返回多样性

## 备注说明:
  - 1. ***如果FQA标准问答对很少, 比如少于1w条数据, 建议不要用bert-white, 其与领域数据相关, 数据量太小会极大降低泛化性***;
  - 2. 可以考虑small、tiny类小模型, 如果要加速推理;
  - 3. annoy安装于linux必须有c++环境, 如gcc-c++, g++等, 只有gcc的话可以用faiss-cpu
  - 4. 增量更新: 建议问题对增量更新/faiss-annoy索引全量更新

## 模型文件
  - 1. 模型文件采用的是 ""
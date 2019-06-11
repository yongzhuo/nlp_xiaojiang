# run
  * 0.数据采用的是webank比赛数据，项目中只有部分几条，
  * 1.谷歌预训练好的模型chinese_L-12_H-768_A-12需要存到Data/chinese_L-12_H-768_A-12目录下
  *   需要的前往需要的前往链接: https://pan.baidu.com/s/1I3vydhmFEQ9nuPG2fDou8Q 提取码: rket
  *   找到webank.rar压缩包，下载覆盖工程目录Data/corpus/webnk文件夹就好；chinese_L-12_H-768_A-12覆盖Data/chinese_L-12_H-768_A-12就好
  * 1.训练
  *    python   keras_bert_classify_bi_lstm.py        或者    python   keras_bert_classify_text_cnn.py
  * 2.测试（__main__下面注释掉train()和predict(), 改为tet()就好，predict()同理）
  *    python   keras_bert_classify_bi_lstm.py
  * 3.预测（__main__下面注释掉train()和tet()，放开predict()）
  *    python   keras_bert_classify_bi_lstm.py

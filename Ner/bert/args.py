# bi-lstm
return_sequences = True
use_cudnn_cell = True
use_lstm = True
use_crf = True
is_training = True

loss = 'categorical_crossentropy'
metrics =  ['accuracy'] # 'crf_loss' # ['accuracy']
activation =  'relu' # 'relu'
optimizers = 'adam'
learning_rate = 1e-3
epsilon = 1e-9
embedding_dim = 768
keep_prob = 0.5
units = 256
decay = 0.0
label = 7
l2 = 0.032

epochs = 320
batch_size = 16
path_save_model = 'models/bilstm/bert_ner_bilstm_no_12_config.h5'
path_tag_li = 'models/bilstm/tag_l_i.pkl'

# gpu使用率
gpu_memory_fraction = 0.32

# ner当然是所有层都会提取啦，句向量默认取倒数第二层的输出值作为句向量
layer_indexes = [i+1 for i in range(13)] # [-2]

# 序列的最大程度
max_seq_len = 50

# bi-lstm
return_sequences = True
use_cudnn_cell = True
use_lstm = True

loss = 'categorical_crossentropy'
metrics =  ['accuracy']
activation =  'softmax'
optimizers = 'adam'
learning_rate = 1e-3
embedding_dim = 768
keep_prob = 0.5
units = 256
decay = 0.0
label = 2
l2 = 0.32

epochs = 100
batch_size = 128
path_save_model = 'model_webank_tdt/bert_avt_cnn.h5' # 'bert_bi_lstm_pair.h5'

# text-cnn
filters = [3, 4, 5]
num_filters = 512



# gpu使用率
gpu_memory_fraction = 0.3

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = 98

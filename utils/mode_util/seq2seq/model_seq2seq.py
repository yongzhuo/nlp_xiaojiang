"""
sequence to sequence Model

Code from: QHDuan(2018-02-05) url: https://github.com/qhduan/just_another_seq2seq

官方文档的意思好像是time_major=True的情况下会快一点
https://www.tensorflow.org/tutorials/seq2seq
不过现在代码都在time_major=False上

test on tensorflow == 1.4.1
seq2seq:  https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/contrib/seq2seq
Code was borrow heavily from:  https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py
Another wonderful example is:  https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm
Official sequence2sequence project:  https://github.com/tensorflow/nmt
Another official sequence2sequence model:  https://github.com/tensorflow/tensor2tensor
"""
from utils.mode_util.seq2seq.data_utils import _get_embed_device
from utils.mode_util.seq2seq.word_sequence import WordSequence

from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import LuongAttention
from tensorflow.contrib.rnn import ResidualWrapper
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import seq2seq
from tensorflow import layers
import tensorflow as tf
import numpy as np


class SequenceToSequence(object):
    """SequenceToSequence Model

    基本流程
    __init__ 基本参数保存，验证参数合法性
        build_model 开始构建整个模型
            init_placeholders 初始化一些tensorflow的变量占位符
            build_encoder 初始化编码器
                build_single_cell
                    build_encoder_cell
            build_decoder 初始化解码器
                build_single_cell
                    build_decoder_cell
            init_optimizer 如果是在训练模式则初始化优化器
    train 训练一个batch的数据
    predict 预测一个batch的数据
    """

    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 batch_size=32,
                 embedding_size=300,
                 mode='train',
                 hidden_units=256,
                 depth=1,
                 beam_width=1,
                 cell_type='lstm',
                 dropout=0.2,
                 use_dropout=False,
                 use_residual=False,
                 optimizer='adam',
                 learning_rate=1e-3,
                 min_learning_rate=1e-6,
                 decay_steps=500000,
                 max_gradient_norm=5.0,
                 max_decode_step=None,
                 attention_type='Bahdanau',
                 bidirectional=False,
                 time_major=False,
                 seed=0,
                 parallel_iterations=None,
                 share_embedding=False,
                 pretrained_embedding=False):
        """保存参数变量，开始构建整个模型
        Args:
            input_vocab_size: 输入词表大小
            target_vocab_size: 输出词表大小
            batch_size: 数据batch的大小
            embedding_size, 输入词表与输出词表embedding的维度
            mode: 取值为 train 或者 decode，训练模式或者预测模式
            hidden_units:
                RNN模型的中间层大小，encoder和decoder层相同
                如果encoder层是bidirectional的话，decoder层是双倍大小
            depth: encoder和decoder的rnn层数
            beam_width:
                beam_width是beamsearch的超参，用于解码
                如果大于0则使用beamsearch，小于等于0则不使用
            cell_type: rnn神经元类型，lstm 或者 gru
            dropout: dropout比例，取值 [0, 1)
            use_dropout: 是否使用dropout
            use_residual:# 是否使用residual
            optimizer: 优化方法， adam, adadelta, sgd, rmsprop, momentum
            learning_rate: 学习率
            max_gradient_norm: 梯度正则剪裁的系数
            max_decode_step:
                最大的解码长度，可以是很大的整数，默认是None
                None的情况下默认是encoder输入最大长度的 4 倍
            attention_type: 'Bahdanau' or 'Luong' 不同的 attention 类型
            bidirectional: encoder 是否为双向
            time_major:
                是否在“计算过程”中使用时间为主的批量数据
                注意，改变这个参数并不要求改变输入数据的格式
                输入数据的格式为 [batch_size, time_step] 是一个二维矩阵
                time_step是句子长度
                经过 embedding 之后，数据会变为
                [batch_size, time_step, embedding_size]
                这是一个三维矩阵（或者三维张量Tensor）
                这样的数据格式是 time_major=False 的
                如果设置 time_major=True 的话，在部分计算的时候，会把矩阵转置为
                [time_step, batch_size, embedding_size]
                也就是 time_step 是第一维，所以叫 time_major
                TensorFlow官方文档认为time_major=True会比较快
            seed: 一些层间操作的随机数 seed 设置
            parallel_iterations:
                dynamic_rnn 和 dynamic_decode 的并行数量
                如果要取得可重复结果，在有dropout的情况下，应该设置为
            share_embedding:
                如果为True，那么encoder和decoder就会公用一个embedding
        """

        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.depth = depth
        self.cell_type = cell_type.lower()
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.attention_type = attention_type
        self.mode = mode
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_steps = decay_steps
        self.max_gradient_norm = max_gradient_norm
        self.keep_prob = 1.0 - dropout
        self.bidirectional = bidirectional
        self.seed = seed
        self.pretrained_embedding = pretrained_embedding
        if isinstance(parallel_iterations, int):
            self.parallel_iterations = parallel_iterations
        else:  # if parallel_iterations is None:
            self.parallel_iterations = batch_size
        self.time_major = time_major
        self.share_embedding = share_embedding
        # Initialize encoder_embeddings to have variance=1.
        # sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        # self.initializer = tf.random_uniform_initializer(
        #     -sqrt3, sqrt3, dtype=tf.float32
        # )
        self.initializer = tf.random_uniform_initializer(
            -0.05, 0.05, dtype=tf.float32
        )
        # self.initializer = None

        assert self.cell_type in ('gru', 'lstm'), \
            'cell_type 应该是 GRU 或者 LSTM'

        if share_embedding:
            assert input_vocab_size == target_vocab_size, \
                '如果打开 share_embedding，两个vocab_size必须一样'

        assert mode in ('train', 'decode'), \
            'mode 必须是 "train" 或 "decode" 而不是 "{}"'.format(mode)

        assert dropout >= 0.0 and dropout < 1.0, '0 <= dropout < 1'

        assert attention_type.lower() in ('bahdanau', 'luong'), \
            '''attention_type 必须是 "bahdanau" 或 "luong" 而不是 "{}"
            '''.format(attention_type)

        assert beam_width < target_vocab_size, \
            'beam_width {} 应该小于 target vocab size {}'.format(
                beam_width, target_vocab_size
            )

        self.keep_prob_placeholder = tf.placeholder(
            tf.float32,
            shape=[],
            name='keep_prob'
        )

        self.global_step = tf.Variable(
            0, trainable=False, name='global_step'
        )

        self.use_beamsearch_decode = False
        self.beam_width = beam_width
        self.use_beamsearch_decode = True if self.beam_width > 0 else False
        self.max_decode_step = max_decode_step

        assert self.optimizer.lower() in \
               ('adadelta', 'adam', 'rmsprop', 'momentum', 'sgd'), \
            'optimizer 必须是下列之一： adadelta, adam, rmsprop, momentum, sgd'

        self.build_model()

    def build_model(self):
        """构建整个模型
        分别构建
        编码器（encoder）
        解码器（decoder）
        优化器（只在训练时构建，optimizer）
        """
        self.init_placeholders()
        encoder_outputs, encoder_state = self.build_encoder()
        self.build_decoder(encoder_outputs, encoder_state)

        if self.mode == 'train':
            self.init_optimizer()

        self.saver = tf.train.Saver()

    def init_placeholders(self):
        """初始化训练、预测所需的变量
        """

        self.add_loss = tf.placeholder(
            dtype=tf.float32,
            name='add_loss'
        )

        # 编码器输入，shape=(batch_size, time_step)
        # 有 batch_size 句话，每句话是最大长度为 time_step 的 index 表示
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size, None),
            name='encoder_inputs'
        )

        # 编码器长度输入，shape=(batch_size, 1)
        # 指的是 batch_size 句话每句话的长度
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size,),
            name='encoder_inputs_length'
        )

        if self.mode == 'train':
            # 训练模式

            # 解码器输入，shape=(batch_size, time_step)
            # 注意，会默认里面已经在每句结尾包含 <EOS>
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32,
                shape=(self.batch_size, None),
                name='decoder_inputs'
            )

            # 解码器输入的reward，用于强化学习训练，shape=(batch_size, time_step)
            self.rewards = tf.placeholder(
                dtype=tf.float32,
                shape=(self.batch_size, 1),
                name='rewards'
            )

            # 解码器长度输入，shape=(batch_size,)
            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32,
                shape=(self.batch_size,),
                name='decoder_inputs_length'
            )

            self.decoder_start_token = tf.ones(
                shape=(self.batch_size, 1),
                dtype=tf.int32
            ) * WordSequence.START

            # 实际训练的解码器输入，实际上是 start_token + decoder_inputs
            self.decoder_inputs_train = tf.concat([
                self.decoder_start_token,
                self.decoder_inputs
            ], axis=1)

    def build_single_cell(self, n_hidden, use_residual):
        """构建一个单独的rnn cell
        Args:
            n_hidden: 隐藏层神经元数量
            use_residual: 是否使用residual wrapper
        """

        if self.cell_type == 'gru':
            cell_type = GRUCell
        else:
            cell_type = LSTMCell

        cell = cell_type(n_hidden)

        if self.use_dropout:
            cell = DropoutWrapper(
                cell,
                dtype=tf.float32,
                output_keep_prob=self.keep_prob_placeholder,
                seed=self.seed
            )

        if use_residual:
            cell = ResidualWrapper(cell)

        return cell

    def build_encoder_cell(self):
        """构建一个单独的编码器cell
        """
        return MultiRNNCell([
            self.build_single_cell(
                self.hidden_units,
                use_residual=self.use_residual
            )
            for _ in range(self.depth)
        ])

    def feed_embedding(self, sess, encoder=None, decoder=None):
        """加载预训练好的embedding
        """
        assert self.pretrained_embedding, \
            '必须开启pretrained_embedding才能使用feed_embedding'
        assert encoder is not None or decoder is not None, \
            'encoder 和 decoder 至少得输入一个吧大佬！'

        if encoder is not None:
            sess.run(self.encoder_embeddings_init,
                     {self.encoder_embeddings_placeholder: encoder})

        if decoder is not None:
            sess.run(self.decoder_embeddings_init,
                     {self.decoder_embeddings_placeholder: decoder})

    def build_encoder(self):
        """构建编码器
        """
        # print("构建编码器")
        with tf.variable_scope('encoder'):
            # 构建 encoder_cell
            encoder_cell = self.build_encoder_cell()

            # 编码器的embedding
            with tf.device(_get_embed_device(self.input_vocab_size)):

                # 加载训练好的embedding
                if self.pretrained_embedding:

                    self.encoder_embeddings = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=(self.input_vocab_size, self.embedding_size)
                        ),
                        trainable=True,
                        name='embeddings'
                    )
                    self.encoder_embeddings_placeholder = tf.placeholder(
                        tf.float32,
                        (self.input_vocab_size, self.embedding_size)
                    )
                    self.encoder_embeddings_init = \
                        self.encoder_embeddings.assign(
                            self.encoder_embeddings_placeholder)

                else:
                    self.encoder_embeddings = tf.get_variable(
                        name='embedding',
                        shape=(self.input_vocab_size, self.embedding_size),
                        initializer=self.initializer,
                        dtype=tf.float32
                    )

            # embedded之后的输入 shape = (batch_size, time_step, embedding_size)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings,
                ids=self.encoder_inputs
            )

            if self.use_residual:
                self.encoder_inputs_embedded = \
                    layers.dense(self.encoder_inputs_embedded,
                                 self.hidden_units,
                                 use_bias=False,
                                 name='encoder_residual_projection')

            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]

            inputs = self.encoder_inputs_embedded
            if self.time_major:
                inputs = tf.transpose(inputs, (1, 0, 2))

            if not self.bidirectional:
                # 单向 RNN
                (
                    encoder_outputs,
                    encoder_state
                ) = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True
                )
            else:
                # 双向 RNN 比较麻烦
                encoder_cell_bw = self.build_encoder_cell()
                (
                    (encoder_fw_outputs, encoder_bw_outputs),
                    (encoder_fw_state, encoder_bw_state)
                ) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=encoder_cell,
                    cell_bw=encoder_cell_bw,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True
                )

                # 首先合并两个方向 RNN 的输出
                encoder_outputs = tf.concat(
                    (encoder_fw_outputs, encoder_bw_outputs), 2)

                encoder_state = []
                for i in range(self.depth):
                    encoder_state.append(encoder_fw_state[i])
                    encoder_state.append(encoder_bw_state[i])
                encoder_state = tuple(encoder_state)

            return encoder_outputs, encoder_state

    def build_decoder_cell(self, encoder_outputs, encoder_state):
        """构建解码器cell"""

        encoder_inputs_length = self.encoder_inputs_length
        batch_size = self.batch_size

        if self.bidirectional:
            encoder_state = encoder_state[-self.depth:]

        if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, (1, 0, 2))

        # 使用 BeamSearchDecoder 的时候，必须根据 beam_width 来成倍的扩大一些变量
        # encoder_outputs, encoder_state, encoder_inputs_length
        # needs to be tiled so that:
        # [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if self.use_beamsearch_decode:
            encoder_outputs = seq2seq.tile_batch(
                encoder_outputs, multiplier=self.beam_width)
            encoder_state = seq2seq.tile_batch(
                encoder_state, multiplier=self.beam_width)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)
            # 如果使用了 beamsearch 那么输入应该是 beam_width 倍于 batch_size 的
            batch_size *= self.beam_width

        # 下面是两种不同的 Attention 机制
        if self.attention_type.lower() == 'luong':
            # 'Luong' style attention: https://arxiv.org/abs/1508.04025
            self.attention_mechanism = LuongAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length
            )
        else:  # Default Bahdanau
            # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
            self.attention_mechanism = BahdanauAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length
            )

        # Building decoder_cell
        cell = MultiRNNCell([
            self.build_single_cell(
                self.hidden_units,
                use_residual=self.use_residual
            )
            for _ in range(self.depth)
        ])

        # 在非训练（预测）模式，并且没开启 beamsearch 的时候，打开 attention 历史信息
        alignment_history = (
            self.mode != 'train' and not self.use_beamsearch_decode
        )

        def cell_input_fn(inputs, attention):
            """根据attn_input_feeding属性来判断是否在attention计算前进行一次投影计算
            """
            if not self.use_residual:
                return array_ops.concat([inputs, attention], -1)

            attn_projection = layers.Dense(self.hidden_units,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')
            return attn_projection(array_ops.concat([inputs, attention], -1))

        cell = AttentionWrapper(
            cell=cell,
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_units,
            alignment_history=alignment_history,
            cell_input_fn=cell_input_fn,
            name='Attention_Wrapper')

        # 空状态
        decoder_initial_state = cell.zero_state(
            batch_size, tf.float32)

        # 传递encoder状态
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=encoder_state)

        # if self.use_beamsearch_decode:
        #     decoder_initial_state = seq2seq.tile_batch(
        #         decoder_initial_state, multiplier=self.beam_width)

        return cell, decoder_initial_state

    def build_decoder(self, encoder_outputs, encoder_state):
        """构建解码器
        """
        with tf.variable_scope('decoder') as decoder_scope:
            # Building decoder_cell and decoder_initial_state
            (
                self.decoder_cell,
                self.decoder_initial_state
            ) = self.build_decoder_cell(encoder_outputs, encoder_state)

            # 解码器embedding
            with tf.device(_get_embed_device(self.target_vocab_size)):
                if self.share_embedding:
                    self.decoder_embeddings = self.encoder_embeddings
                elif self.pretrained_embedding:

                    self.decoder_embeddings = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=(self.target_vocab_size,
                                   self.embedding_size)
                        ),
                        trainable=True,
                        name='embeddings'
                    )
                    self.decoder_embeddings_placeholder = tf.placeholder(
                        tf.float32,
                        (self.target_vocab_size, self.embedding_size)
                    )
                    self.decoder_embeddings_init = \
                        self.decoder_embeddings.assign(
                            self.decoder_embeddings_placeholder)
                else:
                    self.decoder_embeddings = tf.get_variable(
                        name='embeddings',
                        shape=(self.target_vocab_size, self.embedding_size),
                        initializer=self.initializer,
                        dtype=tf.float32
                    )

            self.decoder_output_projection = layers.Dense(
                self.target_vocab_size,
                dtype=tf.float32,
                use_bias=False,
                name='decoder_output_projection'
            )

            if self.mode == 'train':
                # decoder_inputs_embedded:
                # [batch_size, max_time_step + 1, embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings,
                    ids=self.decoder_inputs_train
                )

                # Helper to feed inputs for training:
                # read inputs from dense ground truth vectors
                inputs = self.decoder_inputs_embedded

                if self.time_major:
                    inputs = tf.transpose(inputs, (1, 0, 2))

                training_helper = seq2seq.TrainingHelper(
                    inputs=inputs,
                    sequence_length=self.decoder_inputs_length,
                    time_major=self.time_major,
                    name='training_helper'
                )

                # 训练的时候不在这里应用 output_layer
                # 因为这里会每个 time_step 的进行 output_layer 的投影计算，比较慢
                # 注意这个trick要成功必须设置 dynamic_decode 的 scope 参数
                training_decoder = seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=training_helper,
                    initial_state=self.decoder_initial_state,
                    # output_layer=self.decoder_output_projection
                )

                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(
                    self.decoder_inputs_length
                )

                # decoder_outputs_train: BasicDecoderOutput
                #     namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_output:
                #     if output_time_major=False:
                #         [batch_size, max_time_step + 1, num_decoder_symbols]
                #     if output_time_major=True:
                #         [max_time_step + 1, batch_size, num_decoder_symbols]
                # decoder_outputs_train.sample_id: [batch_size], tf.int32

                (
                    outputs,
                    self.final_state,  # contain attention
                    _  # self.final_sequence_lengths
                ) = seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=self.time_major,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True,
                    scope=decoder_scope
                )

                # More efficient to do the projection
                # on the batch-time-concatenated tensor
                # logits_train:
                # [batch_size, max_time_step + 1, num_decoder_symbols]
                # 训练的时候一次性对所有的结果进行 output_layer 的投影运算
                # 官方NMT库说这样能提高10~20%的速度
                # 实际上我提高的速度会更大
                self.decoder_logits_train = self.decoder_output_projection(
                    outputs.rnn_output
                )

                # masks: masking for valid and padded time steps,
                # [batch_size, max_time_step + 1]
                self.masks = tf.sequence_mask(
                    lengths=self.decoder_inputs_length,
                    maxlen=max_decoder_length,
                    dtype=tf.float32, name='masks'
                )

                # Computes per word average cross-entropy over a batch
                # Internally calls
                # 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default

                decoder_logits_train = self.decoder_logits_train
                if self.time_major:
                    decoder_logits_train = tf.transpose(decoder_logits_train,
                                                        (1, 0, 2))

                self.decoder_pred_train = tf.argmax(
                    decoder_logits_train, axis=-1,
                    name='decoder_pred_train')

                # 下面的一些变量用于特殊的学习训练
                # 自定义rewards，其实我这里是修改了masks
                # train_entropy = cross entropy
                self.train_entropy = \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.decoder_inputs,
                        logits=decoder_logits_train)

                self.masks_rewards = self.masks * self.rewards

                self.loss_rewards = seq2seq.sequence_loss(
                    logits=decoder_logits_train,
                    targets=self.decoder_inputs,
                    weights=self.masks_rewards,
                    average_across_timesteps=True,
                    average_across_batch=True,
                )

                self.loss = seq2seq.sequence_loss(
                    logits=decoder_logits_train,
                    targets=self.decoder_inputs,
                    weights=self.masks,
                    average_across_timesteps=True,
                    average_across_batch=True,
                )

                self.loss_add = self.loss + self.add_loss

            elif self.mode == 'decode':
                # 预测模式，非训练

                start_tokens = tf.tile(
                    [WordSequence.START],
                    [self.batch_size]
                )
                end_token = WordSequence.END

                def embed_and_input_proj(inputs):
                    """输入层的投影层wrapper
                    """
                    return tf.nn.embedding_lookup(
                        self.decoder_embeddings,
                        inputs
                    )

                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding:
                    # uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(
                        start_tokens=start_tokens,
                        end_token=end_token,
                        embedding=embed_and_input_proj
                    )
                    # Basic decoder performs greedy decoding at each time step
                    # print("building greedy decoder..")
                    inference_decoder = seq2seq.BasicDecoder(
                        cell=self.decoder_cell,
                        helper=decoding_helper,
                        initial_state=self.decoder_initial_state,
                        output_layer=self.decoder_output_projection
                    )
                else:
                    # Beamsearch is used to approximately
                    # find the most likely translation
                    # print("building beamsearch decoder..")
                    inference_decoder = BeamSearchDecoder(
                        cell=self.decoder_cell,
                        embedding=embed_and_input_proj,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=self.decoder_initial_state,
                        beam_width=self.beam_width,
                        output_layer=self.decoder_output_projection,
                    )

                # For GreedyDecoder, return
                # decoder_outputs_decode: BasicDecoderOutput instance
                #     namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_decode.rnn_output:
                # if output_time_major=False:
                #     [batch_size, max_time_step, num_decoder_symbols]
                # if output_time_major=True
                #     [max_time_step, batch_size, num_decoder_symbols]
                # decoder_outputs_decode.sample_id:
                # if output_time_major=False
                #     [batch_size, max_time_step], tf.int32
                # if output_time_major=True
                #     [max_time_step, batch_size], tf.int32

                # For BeamSearchDecoder, return
                # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
                #     namedtuple(predicted_ids, beam_search_decoder_output)
                # decoder_outputs_decode.predicted_ids:
                # if output_time_major=False:
                #     [batch_size, max_time_step, beam_width]
                # if output_time_major=True
                #     [max_time_step, batch_size, beam_width]
                # decoder_outputs_decode.beam_search_decoder_output:
                #     BeamSearchDecoderOutput instance
                #     namedtuple(scores, predicted_ids, parent_ids)

                # 官方文档提到的一个潜在的最大长度选择
                # 我这里改为 * 4
                # maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)
                # https://www.tensorflow.org/tutorials/seq2seq

                if self.max_decode_step is not None:
                    max_decode_step = self.max_decode_step
                else:
                    # 默认 4 倍输入长度的输出解码
                    max_decode_step = tf.round(tf.reduce_max(
                        self.encoder_inputs_length) * 4)

                (
                    self.decoder_outputs_decode,
                    self.final_state,
                    _  # self.decoder_outputs_length_decode
                ) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=self.time_major,
                    # impute_finished=True,	# error occurs
                    maximum_iterations=max_decode_step,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True,
                    scope=decoder_scope
                ))

                if not self.use_beamsearch_decode:
                    # decoder_outputs_decode.sample_id:
                    #     [batch_size, max_time_step]
                    # Or use argmax to find decoder symbols to emit:
                    # self.decoder_pred_decode = tf.argmax(
                    #     self.decoder_outputs_decode.rnn_output,
                    #     axis=-1, name='decoder_pred_decode')

                    # Here, we use expand_dims to be compatible with
                    # the result of the beamsearch decoder
                    # decoder_pred_decode:
                    #     [batch_size, max_time_step, 1] (output_major=False)

                    # self.decoder_pred_decode = tf.expand_dims(
                    #     self.decoder_outputs_decode.sample_id,
                    #     -1
                    # )

                    dod = self.decoder_outputs_decode
                    self.decoder_pred_decode = dod.sample_id

                    if self.time_major:
                        self.decoder_pred_decode = tf.transpose(
                            self.decoder_pred_decode, (1, 0))

                else:
                    # Use beam search to approximately
                    # find the most likely translation
                    # decoder_pred_decode:
                    # [batch_size, max_time_step, beam_width] (output_major=False)
                    self.decoder_pred_decode = \
                        self.decoder_outputs_decode.predicted_ids

                    if self.time_major:
                        self.decoder_pred_decode = tf.transpose(
                            self.decoder_pred_decode, (1, 0, 2))

                    self.decoder_pred_decode = tf.transpose(
                        self.decoder_pred_decode,
                        perm=[0, 2, 1])
                    dod = self.decoder_outputs_decode
                    self.beam_prob = dod.beam_search_decoder_output.scores

    def save(self, sess, save_path='model.ckpt'):
        """保存模型"""
        self.saver.save(sess, save_path=save_path)

    def load(self, sess, save_path='model.ckpt'):
        """读取模型"""
        print('try load model from', save_path)
        self.saver.restore(sess, save_path)

    def init_optimizer(self):
        """初始化优化器
        支持的方法有 sgd, adadelta, adam, rmsprop, momentum
        """

        # 学习率下降算法
        learning_rate = tf.train.polynomial_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.min_learning_rate,
            power=0.5
        )
        self.current_learning_rate = learning_rate

        # 设置优化器,合法的优化器如下
        # 'adadelta', 'adam', 'rmsprop', 'momentum', 'sgd'
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer.lower() == 'momentum':
            self.opt = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9)
        elif self.optimizer.lower() == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss, trainable_params)
        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        # Update the model
        self.updates = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step)

        # 使用包括rewards的loss进行更新
        # 是特殊学习的一部分
        gradients = tf.gradients(self.loss_rewards, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        self.updates_rewards = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step)

        # 添加 self.loss_add 的 update
        gradients = tf.gradients(self.loss_add, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        self.updates_add = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step)

    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, decode):
        """检查输入变量，并返回input_feed

        我们首先会把数据编码，例如把“你好吗”，编码为[0, 1, 2]
        多个句子组成一个batch，共同训练，例如一个batch_size=2，那么训练矩阵就可能是
        encoder_inputs = [
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]
        它所代表的可能是：[['我', '是', '帅', '哥'], ['你', '好', '啊', '</s>']]
        注意第一句的真实长度是 4，第二句只有 3（最后的</s>是一个填充数据）

        那么：
        encoder_inputs_length = [4, 3]
        来代表输入整个batch的真实长度
        注意，为了符合算法要求，每个batch的句子必须是长度降序的，也就是说你输入一个
        encoder_inputs_length = [1, 10] 这样是错误的，必须在输入前排序到
        encoder_inputs_length = [10, 1] 这样才行

        decoder_inputs 和 decoder_inputs_length 所代表的含义差不多

        Args:
            encoder_inputs:
                一个整形二维矩阵 [batch_size, max_source_time_steps]
            encoder_inputs_length:
                一个整形向量 [batch_size]
                每个维度是encoder句子的真实长度
            decoder_inputs:
                一个整形矩阵 [batch_size, max_target_time_steps]
            decoder_inputs_length:
                一个整形向量 [batch_size]
                每个维度是decoder句子的真实长度
            decode: 用来指示正在训练模式(decode=False)还是预测模式(decode=True)
        Returns:
            tensorflow所操作需要的input_feed，包括
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length
        """

        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError(
                "encoder_inputs和encoder_inputs_length的第一维度必须一致 "
                "这一维度是batch_size, %d != %d" % (
                    input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError(
                    "encoder_inputs和decoder_inputs的第一维度必须一致 "
                    "这一维度是batch_size, %d != %d" % (
                        input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError(
                    "edeoder_inputs和decoder_inputs_length的第一维度必须一致 "
                    "这一维度是batch_size, %d != %d" % (
                        target_batch_size, decoder_inputs_length.shape[0]))

        input_feed = {}

        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed

    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length,
              rewards=None, return_lr=False,
              loss_only=False, add_loss=None):
        """训练模型"""

        # 输入
        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length,
            False
        )

        # 设置 dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        if loss_only:
            # 输出
            return sess.run(self.loss, input_feed)

        if add_loss is not None:
            input_feed[self.add_loss.name] = add_loss
            output_feed = [
                self.updates_add, self.loss_add,
                self.current_learning_rate]
            _, cost, lr = sess.run(output_feed, input_feed)

            if return_lr:
                return cost, lr

            return cost

        if rewards is not None:
            input_feed[self.rewards.name] = rewards
            output_feed = [
                self.updates_rewards, self.loss_rewards,
                self.current_learning_rate]
            _, cost, lr = sess.run(output_feed, input_feed)

            if return_lr:
                return cost, lr
            return cost

        output_feed = [
            self.updates, self.loss,
            self.current_learning_rate]
        _, cost, lr = sess.run(output_feed, input_feed)

        if return_lr:
            return cost, lr

        return cost

    def get_encoder_embedding(self, sess, encoder_inputs):
        """获取经过embedding的encoder_inputs"""
        input_feed = {
            self.encoder_inputs.name: encoder_inputs
        }
        emb = sess.run(self.encoder_inputs_embedded, input_feed)
        return emb

    def entropy(self, sess, encoder_inputs, encoder_inputs_length,
                decoder_inputs, decoder_inputs_length):
        """获取针对一组输入输出的entropy
        相当于在计算P(target|source)
        """
        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length,
            False
        )
        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = [self.train_entropy, self.decoder_pred_train]
        entropy, logits = sess.run(output_feed, input_feed)
        return entropy, logits

    def predict(self, sess,
                encoder_inputs,
                encoder_inputs_length,
                attention=False):
        """预测输出"""

        # 输入
        input_feed = self.check_feeds(encoder_inputs,
                                      encoder_inputs_length, None, None, True)

        input_feed[self.keep_prob_placeholder.name] = 1.0

        # Attention 输出
        if attention:
            assert not self.use_beamsearch_decode, \
                'Attention 模式不能打开 BeamSearch'

            pred, atten = sess.run([
                self.decoder_pred_decode,
                self.final_state.alignment_history.stack()
            ], input_feed)

            return pred, atten

        # BeamSearch 模式输出
        if self.use_beamsearch_decode:
            pred, beam_prob = sess.run([
                self.decoder_pred_decode, self.beam_prob
            ], input_feed)
            beam_prob = np.mean(beam_prob, axis=1)

            pred = pred[0]
            return pred

            # ret = []
            # for i in range(encoder_inputs.shape[0]):
            #     ret.append(pred[i * self.beam_width])
            # return np.array(ret)
            #

        # 普通（Greedy）模式输出
        pred, = sess.run([
            self.decoder_pred_decode
        ], input_feed)

        return pred

import tensorflow as tf
import numpy as np

from utils import DataLoader 

SRC_TRAIN_DATA = 'data\\source.txt'
TRG_TRAIN_DATA = 'data\\target.txt'
EMB_DATA = 'data\\vocab.emb'
CHECKPOINT_PATH = 'models\\v1\\ckpt'
RESTORE_PATH = 'models\\v1'
HIDDEN_SIZE = 300
EMBEDDING_SIZE = 300 # 使用的函数tf.contrib.rnn.MultiRNNcell会自动将累加的LSTM的参数设为相同的shape,因为每层的Wo的维度不能单独设置，所以所有的Wi=Wo=[emb_size, hidden_size]，并且输出以hidden_size为单位
NUM_LAYER = 2
VOCAB_SIZE = 4950
BATCH_SIZE = 1000
NUM_EPOCH = 5
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
MAX_LEN = 30 
SOS_ID = 1
EOS_ID = 2

class NMTModel(object):

    def __init__(self):
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, name='basic_lstm_cell') for _ in range(NUM_LAYER)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, name='basic_lstm_cell') for _ in range(NUM_LAYER)])

        self.embedding = tf.get_variable(name="embedding", dtype=tf.float32, shape=(VOCAB_SIZE, EMBEDDING_SIZE))

        self.softmax_weight = tf.transpose(self.embedding)
        self.softmax_bias = tf.get_variable("softmax_bias", [VOCAB_SIZE])

    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        np.set_printoptions(threshold=np.nan) 
        # 获取当前batch的大小
        batch_size = tf.shape(src_input)[0]

        src_emb = tf.nn.embedding_lookup(self.embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.embedding, trg_input)

        # dropout
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb, src_size, dtype=tf.float32)
        
        # dec_outputs的维度是[batch_size, max_time, EMBEDDING_SIZE]
        with tf.variable_scope("decoder"):
            dec_outputs, _ = tf.nn.dynamic_rnn(self.dec_cell, trg_emb, trg_size, initial_state=enc_state)

        # 计算每一步的log perplexity
        # reshape就是把所有batch的回答都连接起来成一个二维矩阵，每一行代表每个词的词向量
        output = tf.reshape(dec_outputs, [-1, EMBEDDING_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        # 此处的reshape就是将trg_label全部展开，一个元素代表一个字
        # sparse_softmax_cross_entropy_with_logits就是用每一个字的代号数字去和logits（词汇表各词可能性的概率分布）一起计算交叉熵
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]), logits=logits)

        # 计算损失值时要把填充的无意义的字的损失值记为0
        label_weights = tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        # 对位相乘，填充位置的loss就变为0了
        with tf.name_scope("cost"):
            mask_loss = loss * label_weights
            cost = tf.reduce_mean(mask_loss)
            tf.summary.scalar('cost', cost)
            # cost_per_token = cost / tf.reduce_sum(label_weights)

        trainable_variables = tf.trainable_variables()

        # 控制梯度大小
        # grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
        grads = tf.gradients(cost, trainable_variables)
        # pot = tf.print('loss:', mask_loss, 'src:',src_input, 'trg:', trg_input, 'cost:', cost, summarize=-1)

        # with tf.control_dependencies([pot]):
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        
        return cost, train_op

    def inference(self, src_input):
        # 因为dynamic_rnn要求的是batch的形式，所以整理为batch_size为1的tensor
        # src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        # src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_size = tf.convert_to_tensor([tf.shape(src_input)[1]], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.embedding, src_input)

        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb, src_size, dtype=tf.float32)

        with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
            # 生成一个变长的tensorarray来存储结果
            init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
            # 填入第一个单词SOS作为解码器的输入
            init_array = init_array.write(0, SOS_ID)
            # 构建循环初始状态
            init_loop_var = (enc_state, init_array, 0)

            # 循环解码的结束条件
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID), 
                    tf.less(step, MAX_LEN) ))

            def loop_body(state, trg_ids, step):
                # 读取最后一步输出的单词，并读取其词向量
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.embedding, trg_input)
                # 这里不用dynamic_rnn，只是用dec_cell往前调用一次
                dec_outputs, next_state = self.dec_cell.call(state=state, inputs=trg_emb)
                # 通过概率分布logit计算输出哪个字比较合适
                output = tf.reshape(dec_outputs, [-1, EMBEDDING_SIZE])
                logits = (tf.matmul(output, self.softmax_weight) + self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # 将这一步的输出写入循环状态中
                # pot = tf.print(tf.shape(output), tf.shape(logits),tf.shape(self.softmax_weight), output, logits,next_id, state, src_size, src_input, tf.shape(src_input), src_emb)
                # with tf.control_dependencies([pot]):
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1

            state, trg_ids, step = tf.while_loop(continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()

def run_epoch(session, cost_op, train_op, saver, step, summary_write, merged):
    while True:
        try:
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print("After %d steps, per token cost is %.4f" % (step, cost))
            if step % 100 == 0:
                # tensorboard
                summary = session.run(merged)
                summary_write.add_summary(summary, step)
            if step % 2000 ==0 and step > 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            # 读取到末尾没有其它信息后会抛出这个异常，捕获之后继续下一个run_epoch
            break
    return step

def main():
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    data_loader = DataLoader()

    with tf.variable_scope("nmt_model", reuse=None, initializer=initializer):
        train_model = NMTModel()

    data = data_loader.MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src,src_size), (trg_input, trg_label, trg_size) = iterator.get_next()
    
    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)

    saver = tf.train.Saver()
    step = 0

    # LOSS统计图
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_write = tf.summary.FileWriter("logs/", sess.graph)
        tf.global_variables_initializer().run()
        checkpoint = tf.train.latest_checkpoint(RESTORE_PATH)
        print('从存档重新开始', checkpoint)
        if checkpoint is not None:
            saver.restore(sess, checkpoint)
        for i in range(NUM_EPOCH):
            print("In iterator: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step, summary_write, merged)
        summary_write.close()

if __name__ == "__main__":
    main()

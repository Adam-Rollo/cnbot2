import tensorflow as tf
import pickle
import jieba

MAX_LEN = 30 # 每句话最多30个字
SOS_ID = 1 # Start Symbol的ID为1

SRC_FILE = 'data\\source.txt'
TRG_FILE = 'data\\target.txt'

class DataGenerator:

    def __init__(self, vocab_file = None, embedding_file = None):
        if vocab_file is not None:
            self._LoadVocab(vocab_file, embedding_file)

    def create_chars_dict(self, file_path):
        word_dict = {}
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as source_file:
            for line in source_file:
                for word in line:
                    if word == ' ' or word == '\n':
                        continue
                    if word in word_dict:
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1

        self.chars = [v for v in sorted(word_dict.items(), key=lambda d: d[1], reverse=True)]
        self.chars = list(filter(lambda x: x[1] > 10, self.chars))
        self.chars = ['UNK', 'BOS', 'EOS'] + list(map(lambda x: x[0], self.chars))
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        with open('data\\vocab.pkl', 'wb') as f:
            pickle.dump(self.chars, f)
        print("Saved vocab (vocab size: {:,d})".format(len(self.chars)))

    def _LoadVocab(self, vocab_file, embedding_file):
        '''在另一个项目中，我们已经做好了词汇表以及词向量表
            汉语词汇的数量缩减为八万'''
        with open(vocab_file, 'rb') as f:
            self.chars = pickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

    def MakeFile(self, file_path, cut=False):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as source_file:
            question = []
            line_num = 0
            for line in source_file:
                line_num += 1
                if line_num % 10000 == 0:
                    print(line_num, line)
                    anwser = list(map(lambda x: x + ['2'], question[1:]))
                    question = question[:-1]
                    question = list(map(lambda x:' '.join(x), question))
                    anwser = list(map(lambda x:' '.join(x), anwser))
                    with open(SRC_FILE, 'a+', encoding='utf-8') as source_file:
                        source_file.write('\n'.join(question) + '\n') 

                    with open(TRG_FILE, 'a+', encoding='utf-8') as target_file:
                        target_file.write('\n'.join(anwser) + '\n') 

                    question = []

                if cut:
                    # words = [word.strip() for word in jieba.cut(line, cut_all=False) ]
                    words = [word.strip() for word in line]
                else:
                    words = [word.strip() for word in line.split()]
                ids = list(map(lambda x:str(self.vocab.get(x)) if self.vocab.get(x) else '0', words))
                ids = list(filter(lambda x:x.isdigit() and x!='0', ids))
                if ids:
                    question.append(ids)


class DataLoader:

    def LoadVocab(self, vocab_file):
        with open(vocab_file, 'rb') as f:
            self.chars = pickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        return self.vocab

    def LoadEmb(self, embedding_file):
        with open(embedding_file, 'rb') as f:
            self.embeddings = pickle.load(f)
        return self.embeddings

    def MakeDataset(self, file_path):
        dataset = tf.data.TextLineDataset(file_path)
        # 按空格切开
        dataset = dataset.map(lambda string: tf.string_split([string]).values)
        # Converts each string in the input Tensor to the specified numeric type.
        dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
        # 统计每个句子的单词数量，一起放入dataset
        dataset = dataset.map(lambda x: (x, tf.size(x)))
        return dataset

    def MakeSrcTrgDataset(self, src_path, trg_path, batch_size):
        # 读取问题和答案
        src_data = self.MakeDataset(src_path)
        trg_data = self.MakeDataset(trg_path)
        # zip将两个dataset合并成一个，现在dataset中每一个元素ds由四个张量组成
        # ds[0][0]问题句子,ds[0][1]是问题句子长度,ds[1][0]是回答句子,ds[1][1]是回答句子长度
        dataset = tf.data.Dataset.zip((src_data, trg_data))

        # 过滤掉内容为空（只有eos）或者超过最大长度的句子
        def FilterLength(src_tuple, trg_tuple):
            ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
            src_len_ok = tf.logical_and(tf.greater(src_len, 0), tf.less_equal(src_len, MAX_LEN))
            trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
            return tf.logical_and(src_len_ok, trg_len_ok)
        dataset = dataset.filter(FilterLength)

        # 解码器输入(trg_input)BOS X Y Z，依次得到(trg_label)X Y Z EOS
        def MakeTrgInput(src_tuple, trg_tuple):
            ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
            trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
            return ((src_input, src_len), (trg_input, trg_label, trg_len))
        dataset = dataset.map(MakeTrgInput)

        dataset = dataset.shuffle(10000)

        # 这个格式是dynamic_rnn需要的格式
        padded_shapes = (
            (tf.TensorShape([None]),
            tf.TensorShape([])),
            (tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([])))
        # Pad: If the dimension is unknown (e.g. tf.Dimension(None)), the component will be padded out to the maximum length of all elements in that dimension.
        # batch_size为2时，得到的结果为( (array([...],[...]), array(n,n)), (array([...],[...], array([...],[...]), array(n,n)) )
        batched_dataset = dataset.padded_batch(batch_size, padded_shapes)

        return batched_dataset

    def get_words_id(self, words):
        ids = list(map(lambda x:self.vocab.get(x) if self.vocab.get(x) else 0, words))
        return ids

    def get_words_by_id(self, ids):
        my_dict = {v: k for k, v in self.vocab.items()}
        words = list(map(lambda x:my_dict.get(x) if my_dict.get(x) else '', ids))
        return words

if __name__ == "__main__" :
    gen = DataGenerator("data\\vocab.pkl")
    gen.MakeFile('data\subtitle.corpus', True)
    '''
    gen = DataGenerator()
    gen.create_chars_dict('data\\mline_dialog.conv')
    gen.MakeFile('data\\mline_dialog.conv')
    loader = DataLoader()
    bd = loader.MakeSrcTrgDataset('data\\source.txt', 'data\\target.txt', 50)
    it = bd.make_one_shot_iterator()
    x = it.get_next()
    with tf.Session() as sess:
        print(sess.run(x))
    '''

import tensorflow as tf
import jieba
import os

from utils import DataLoader 
from train import NMTModel

CHECKPOINT_PATH = 'models\\v1\\ckpt-10000'
VOCAB_FILE = 'data\\vocab.pkl'

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

    data_loader = DataLoader()
    data_loader.LoadVocab(VOCAB_FILE)

    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModel()

    question = tf.placeholder(tf.int32, shape=[1,None], name='question')
    res = model.inference(question)

    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, CHECKPOINT_PATH)

        talking = True
        while talking:
            q = input("> ")
            # seg_list = jieba.cut(q, cut_all=False) 
            seg_list = [word for word in q]
            q = data_loader.get_words_id(seg_list)
            anwser = sess.run(res, feed_dict={question:[q]})
            anwser = data_loader.get_words_by_id(anwser)
            print(''.join(anwser[1:-1]))

if __name__ == "__main__":
    main()

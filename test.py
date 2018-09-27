import logging
import os
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import tensorflow as tf
import numpy as np
if __name__ == '__main__':

    #
    # components = np.arange(100).astype(np.int64)
    # # print(components)
    # dataset = tf.contrib.data.Dataset.from_tensor_slices(components)
    # dataset = dataset.group_by_window(key_func=lambda x: x % 2, reduce_func=lambda _, els: els.batch(10),
    #                                   window_size=100)
    # iterator = dataset.make_one_shot_iterator()
    # features = iterator.get_next()
    # sess = tf.Session()
    # print(sess.run(features))
    # program = os.path.basename(sys.argv[0])
    # logger = logging.getLogger(program)
    #
    # logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    # logging.root.setLevel(level=logging.INFO)
    # logger.info("running %s" % ' '.join(sys.argv))
    #
    # # check and process input arguments
    # # if len(sys.argv) < 4:
    # #     print(globals()['__doc__'] % locals())
    # #     sys.exit(1)
    # inp, outp1, outp2 = sys.argv[1:4]
    inp  = "/home/zxp/Codes/python/SRL_RU_2/resource/source1.txt"
    outp1 = "text.model"
    outp2 = "embeddings.vec"

    model = Word2Vec(LineSentence(inp), size=300, window=5, min_count=15,
                     workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
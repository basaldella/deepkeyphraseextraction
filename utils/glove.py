import os
import numpy as np
import logging


def load_glove(glove_dir,size):
    embeddings_index = {}
    f = open(os.path.join(glove_dir, ('glove.6B.%sd.txt' % size)))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    logging.info('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index

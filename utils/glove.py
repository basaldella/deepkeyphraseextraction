import os
import numpy as np
import logging


def load_glove(glove_dir,size):
    embeddings_index = {}
    glove_path = ('glove.6B.%sd.txt' % size)

    logging.debug("Loading GloVe pre-trained embeddings from %s" % glove_path)

    f = open(os.path.join(glove_dir, glove_path))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    logging.debug('Total embeddings found: %s.' % len(embeddings_index))

    return embeddings_index

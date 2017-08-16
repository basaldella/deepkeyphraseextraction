import logging


def log_versions():
    import keras
    logging.info("Keras version %s" % keras.__version__)
    import numpy as np
    logging.info("Numpy version %s" % np.__version__)
    if keras.backend.backend() == 'theano':
        import theano
        logging.info("Theano version %s" % theano.__version__)
    else:
        import tensorflow
        logging.info("Tensorflow version %s" % tensorflow.__version__)

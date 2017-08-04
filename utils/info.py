import logging


def log_versions():
    import theano
    logging.info("Theano version %s" % theano.__version__)
    import keras
    logging.info("Keras version %s" % keras.__version__)
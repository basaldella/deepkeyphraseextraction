import numpy as np
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os

os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(1337)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(7331)

import logging

from keras import layers, regularizers
from keras.models import Model, load_model

from data.datasets import *
from eval import keras_metrics, metrics
from nlp import chunker, tokenizer as tk
from utils import info, preprocessing, postprocessing, plots

# LOGGING CONFIGURATION

logging.basicConfig(
    format='%(asctime)s\t%(levelname)s\t%(message)s',
    level=logging.DEBUG)

info.log_versions()

# END LOGGING CONFIGURATION

# GLOBAL VARIABLES

SAVE_MODEL = True
MODEL_PATH = "../models/answerrnn2.h5"
SHOW_PLOTS = True
SAMPLE_SIZE = -1  # training set will be restricted to SAMPLE_SIZE. Set to -1 to disable
KP_CLASS_WEIGHT = 1.  # weight of positives samples while training the model. NOTE: MUST be a float

# END GLOBAL VARIABLES

# Dataset and hyperparameters for each dataset

DATASET = Kp20k

if DATASET == Semeval2017:
    tokenizer = tk.tokenizers.nltk
    DATASET_FOLDER = "../data/Semeval2017"
    MAX_DOCUMENT_LENGTH = 540  # gl: same as hulth, so it uses same conv; 400 if use dedicated conv
    MAX_VOCABULARY_SIZE = 12000  # gl: was 20000
    MAX_ANSWER_LENGTH = 12  # gl: was 16 or 27
    EMBEDDINGS_SIZE = 1024
    BATCH_SIZE = 256
    PREDICT_BATCH_SIZE = 256
    EPOCHS = 10
elif DATASET == Hulth:
    tokenizer = tk.tokenizers.nltk
    DATASET_FOLDER = "../data/Hulth2003"
    MAX_DOCUMENT_LENGTH = 540
    MAX_VOCABULARY_SIZE = 20000
    MAX_ANSWER_LENGTH = 12
    EMBEDDINGS_SIZE = 50  # gl: was 50
    BATCH_SIZE = 256
    PREDICT_BATCH_SIZE = 2048
    EPOCHS = 9
elif DATASET == Kp20k:
    tokenizer = tk.tokenizers.nltk
    DATASET_FOLDER = "../data/Kp20k"
    MAX_DOCUMENT_LENGTH = 1407
    MAX_VOCABULARY_SIZE = 170000
    MAX_ANSWER_LENGTH = 100
    EMBEDDINGS_SIZE = 300  # gl: was 50
    BATCH_SIZE = 256
    PREDICT_BATCH_SIZE = 2048
    EPOCHS = 9
elif DATASET == Krapivin2009:
    tokenizer = tk.tokenizers.nltk
    DATASET_FOLDER = "../data/Krapivin2009"
    MAX_DOCUMENT_LENGTH = 550  # gl: was 454
    MAX_VOCABULARY_SIZE = 20000
    MAX_ANSWER_LENGTH = 12
    EMBEDDINGS_SIZE = 300  # gl: was 50
    BATCH_SIZE = 256
    PREDICT_BATCH_SIZE = 256
    EPOCHS = 9
else:
    raise NotImplementedError("Can't set the hyperparameters: unknown dataset")

# END PARAMETERS

# Loss function


def cos_distance(y_true, y_pred):
    import keras.backend as K

    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())

    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))

# End loss


logging.info("Loading dataset...")

data = DATASET(DATASET_FOLDER)

train_doc_str, train_answer_str = data.load_train()
test_doc_str, test_answer_str = data.load_test()
val_doc_str, val_answer_str = data.load_validation()

train_doc, train_answer = tk.tokenize_set(train_doc_str, train_answer_str, tokenizer)
test_doc, test_answer = tk.tokenize_set(test_doc_str, test_answer_str, tokenizer)
val_doc, val_answer = tk.tokenize_set(val_doc_str, val_answer_str, tokenizer)

logging.info("Dataset loaded. Generating candidate keyphrases...")

train_candidates = chunker.extract_candidates_from_set(train_doc_str, tokenizer)
test_candidates = chunker.extract_candidates_from_set(test_doc_str, tokenizer)
val_candidates = chunker.extract_candidates_from_set(val_doc_str, tokenizer)

logging.debug("Candidates recall on training set   : %.4f", metrics.recall(train_answer, train_candidates))
logging.debug("Candidates recall on test set       : %.4f", metrics.recall(test_answer, test_candidates))
logging.debug("Candidates recall on validation set : %.4f", metrics.recall(val_answer, val_candidates))

logging.info("Candidates generated. Preprocessing data...")

train_x, train_y, test_x, test_y, val_x, val_y, val_x_b, val_y_b, embedding_matrix, dictionary = preprocessing. \
    prepare_answer_2(train_doc, train_answer, train_candidates,
                     test_doc, test_answer, test_candidates,
                     val_doc, val_answer, val_candidates,
                     max_document_length=MAX_DOCUMENT_LENGTH,
                     max_answer_length=MAX_ANSWER_LENGTH,
                     max_vocabulary_size=MAX_VOCABULARY_SIZE,
                     embeddings_size=EMBEDDINGS_SIZE)

# Finalize the ys: remove one-hot

train_y = np.argmax(train_y, axis=1)
test_y = np.argmax(test_y, axis=1)
val_y = np.argmax(val_y, axis=1)
val_y_b = np.argmax(val_y_b, axis=1)

logging.info("Data preprocessing complete.")

if not SAVE_MODEL or not os.path.isfile(MODEL_PATH):

    # Dataset sampling

    if 0 < SAMPLE_SIZE < np.shape(train_x[0])[0]:

        logging.warning("Training network on %s samples" % SAMPLE_SIZE)
        samples_indices = rn.sample(range(np.shape(train_x[0])[0]), SAMPLE_SIZE)

        train_x_doc_sample = np.zeros((SAMPLE_SIZE, MAX_DOCUMENT_LENGTH))
        train_x_answer_sample = np.zeros((SAMPLE_SIZE, MAX_ANSWER_LENGTH))

        shape_y = list(np.shape(train_y))
        shape_y[0] = SAMPLE_SIZE

        train_y_sample = np.zeros(tuple(shape_y))

        i = 0
        for j in samples_indices:
            train_x_doc_sample[i] = train_x[0][j]
            train_x_answer_sample[i] = train_x[1][j]
            train_y_sample[i] = train_y[j]
            i += 1

        train_x = [train_x_doc_sample, train_x_answer_sample]
        train_y = train_y_sample

        logging.debug("Sampled Training set documents size : %s", np.shape(train_x[0]))
        logging.debug("Sampled Training set answers size   : %s", np.shape(train_x[1]))

    # end sampling.

    # Class weights

    class_weights = {0: 1.,
                     1: KP_CLASS_WEIGHT}

    logging.debug("Building the network...")
    document = layers.Input(shape=(MAX_DOCUMENT_LENGTH,))
    encoded_document = layers.Embedding(np.shape(embedding_matrix)[0],
                                        EMBEDDINGS_SIZE,
                                        weights=[embedding_matrix],
                                        input_length=MAX_DOCUMENT_LENGTH,
                                        trainable=False)(document)

    '''
    # Size of the output layer for a Convolutional Layer
    # (from http://cs231n.github.io/convolutional-networks/)

    # We can compute the spatial size of the output volume as a function of the input volume size (W),
    # the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S),
    # and the amount of zero padding used (P) on the border.

    # You can convince yourself that the correct formula
    # for calculating how many neurons “fit” is given by ((W−F+2P)/S)+1.


    #encoded_document = layers.Bidirectional(
    #    layers.LSTM(int(EMBEDDINGS_SIZE),
    #                activation='hard_sigmoid',
    #                recurrent_activation='hard_sigmoid',
    #                return_sequences=True))\
    #    (encoded_document)
    '''
    # print(encoded_document)  # gl
    '''
    if DATASET == Hulth:
        encoded_document = layers.Conv1D(filters=128, kernel_size=32, strides=4, activation='relu')(encoded_document)
    elif DATASET == Semeval2017:
        encoded_document = layers.Conv1D(filters=128, kernel_size=25, strides=3, activation='relu')(encoded_document)
    '''
    encoded_document = layers.Conv1D(filters=128, kernel_size=32, strides=4, activation='relu')(encoded_document)
    # Size: 131
    encoded_document = layers.MaxPool1D(pool_size=2)(encoded_document)
    encoded_document = layers.Activation('relu')(encoded_document)
    # Size: 65
    encoded_document = layers.Conv1D(filters=128, kernel_size=8, strides=2, activation='relu')(encoded_document)
    # # Size: 29
    encoded_document = layers.MaxPool1D(pool_size=2)(encoded_document)
    encoded_document = layers.Activation('relu')(encoded_document)
    # # Size: 14
    encoded_document = layers.Conv1D(filters=128, kernel_size=4, strides=1, activation='relu')(encoded_document)
    # # Size: 11
    encoded_document = layers.MaxPool1D(pool_size=2)(encoded_document)
    encoded_document = layers.Activation('relu')(encoded_document)
    # # Size: 5
    # encoded_document = layers.TimeDistributed(layers.Dense(10, activation='softmax'))(encoded_document)
    encoded_document = layers.Flatten()(encoded_document)
 
    print((Model(document, encoded_document)).summary())  # was commented

    candidate = layers.Input(shape=(MAX_ANSWER_LENGTH,))
    encoded_candidate = layers.Embedding(np.shape(embedding_matrix)[0],
                                         EMBEDDINGS_SIZE,
                                         weights=[embedding_matrix],
                                         input_length=MAX_ANSWER_LENGTH,
                                         trainable=False)(candidate)
    '''
    #encoded_candidate = layers.Bidirectional(
    #    layers.LSTM(int(EMBEDDINGS_SIZE),
    #                activation='hard_sigmoid',
    #                recurrent_activation='hard_sigmoid',
    #                return_sequences=True))\
    #    (encoded_candidate)
    '''
    encoded_candidate = layers.Conv1D(filters=128, kernel_size=2, activation='relu')(encoded_candidate)
    encoded_candidate = layers.MaxPool1D(pool_size=2)(encoded_candidate)
    '''
    if DATASET == Hulth:
        encoded_candidate = layers.MaxPool1D(pool_size=2)(encoded_candidate)
    elif DATASET == Semeval2017:
        encoded_candidate = layers.MaxPool1D(pool_size=5)(encoded_candidate)
    '''
    encoded_candidate = layers.Activation('relu')(encoded_candidate)
    encoded_candidate = layers.Flatten()(encoded_candidate)
    print((Model(candidate, encoded_candidate)).summary())  # was commented

    prediction = layers.dot([encoded_document, encoded_candidate], axes=-1, normalize=True)

    model = Model([document, candidate], prediction)

    logging.info("Compiling the network...")
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    '''
    #merged = layers.add([encoded_document, encoded_candidate])
    #prediction = layers.Dense(int(EMBEDDINGS_SIZE / 4), activation='relu',kernel_regularizer=regularizers.l2(0.01))(merged)
    #prediction = layers.Dropout(0.25)(prediction)
    #prediction = layers.Dense(2, activation='softmax')(prediction)

    #model = Model([document, candidate], prediction)

    #logging.info("Compiling the network...")
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    '''

    print(model.summary())

    metrics_callback = keras_metrics.MetricsCallbackQA(val_x, val_y, batch_size=PREDICT_BATCH_SIZE)

    logging.info("Fitting the network...")
    history = model.fit(train_x, train_y,
                        validation_data=(val_x_b, val_y_b),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        class_weight=class_weights,
                        callbacks=[metrics_callback])

    if SHOW_PLOTS:
        plots.plot_accuracy(history)
        plots.plot_loss(history)
        plots.plot_prf(metrics_callback)

    if SAVE_MODEL:
        model.save(MODEL_PATH)
        logging.info("Model saved in %s", MODEL_PATH)

else:
    logging.info("Loading existing model from %s...", MODEL_PATH)
    model = load_model(MODEL_PATH)

logging.info("Predicting on test set...")
output = model.predict(x=test_x, verbose=1, batch_size=PREDICT_BATCH_SIZE)
logging.debug("Shape of output array: %s", np.shape(output))

obtained_words = postprocessing.get_answers(test_candidates, test_x, output, dictionary)

precision = metrics.precision(test_answer, obtained_words)
recall = metrics.recall(test_answer, obtained_words)
f1 = metrics.f1(precision, recall)

print("###    Obtained Scores    ###")
print("###     (full dataset)    ###")
print("###")
print("### Precision : %.4f" % precision)
print("### Recall    : %.4f" % recall)
print("### F1        : %.4f" % f1)
print("###                       ###")

keras_precision = keras_metrics.keras_precision_qa(test_y, output)
keras_recall = keras_metrics.keras_recall_qa(test_y, output)
keras_f1 = keras_metrics.keras_f1_qa(test_y, output)

print("###    Obtained Scores    ###")
print("###    (fixed dataset)    ###")
print("###")
print("### Precision : %.4f" % keras_precision)
print("### Recall    : %.4f" % keras_recall)
print("### F1        : %.4f" % keras_f1)
print("###                       ###")

obtained_words_top = postprocessing.get_top_answers(test_candidates, test_x, output, dictionary, 5)

precision_top = metrics.precision(test_answer, obtained_words_top)
recall_top = metrics.recall(test_answer, obtained_words_top)
f1_top = metrics.f1(precision_top, recall_top)

print("###    Obtained Scores    ###")
print("### (full dataset, top 5) ###")
print("###")
print("### Precision : %.4f" % precision_top)
print("### Recall    : %.4f" % recall_top)
print("### F1        : %.4f" % f1_top)
print("###                       ###")

obtained_words_top = postprocessing.get_top_answers(test_candidates, test_x, output, dictionary, 10)

precision_top = metrics.precision(test_answer, obtained_words_top)
recall_top = metrics.recall(test_answer, obtained_words_top)
f1_top = metrics.f1(precision_top, recall_top)

print("###    Obtained Scores    ###")
print("### (full dataset, top 10)###")
print("###")
print("### Precision : %.4f" % precision_top)
print("### Recall    : %.4f" % recall_top)
print("### F1        : %.4f" % f1_top)
print("###                       ###")

obtained_words_top = postprocessing.get_top_answers(test_candidates, test_x, output, dictionary, 15)

precision_top = metrics.precision(test_answer, obtained_words_top)
recall_top = metrics.recall(test_answer, obtained_words_top)
f1_top = metrics.f1(precision_top, recall_top)

print("###    Obtained Scores    ###")
print("### (full dataset, top 15)###")
print("###")
print("### Precision : %.4f" % precision_top)
print("### Recall    : %.4f" % recall_top)
print("### F1        : %.4f" % f1_top)
print("###                       ###")

print("###                       ###")
print("###                       ###")
print("###       STEMMING        ###")
print("###                       ###")
print("###                       ###")

STEM_MODE = metrics.stemMode.both

precision = metrics.precision(test_answer, obtained_words, STEM_MODE)
recall = metrics.recall(test_answer, obtained_words, STEM_MODE)
f1 = metrics.f1(precision, recall)

print("###    Obtained Scores    ###")
print("###     (full dataset)    ###")
print("###")
print("### Precision : %.4f" % precision)
print("### Recall    : %.4f" % recall)
print("### F1        : %.4f" % f1)
print("###                       ###")

clean_words = postprocessing.get_valid_patterns(obtained_words)

precision = metrics.precision(test_answer, clean_words, STEM_MODE)
recall = metrics.recall(test_answer, clean_words, STEM_MODE)
f1 = metrics.f1(precision, recall)

print("###    Obtained Scores    ###")
print("### (full dataset,        ###")
print("###  pos patterns filter) ###")
print("###")
print("### Precision : %.4f" % precision)
print("### Recall    : %.4f" % recall)
print("### F1        : %.4f" % f1)
print("###                       ###")

obtained_words_top = postprocessing.get_top_answers(test_candidates, test_x, output, dictionary, 5)

precision_top = metrics.precision(test_answer, obtained_words_top, STEM_MODE)
recall_top = metrics.recall(test_answer, obtained_words_top, STEM_MODE)
f1_top = metrics.f1(precision_top, recall_top)

print("###    Obtained Scores    ###")
print("### (full dataset, top 5) ###")
print("###")
print("### Precision : %.4f" % precision_top)
print("### Recall    : %.4f" % recall_top)
print("### F1        : %.4f" % f1_top)
print("###                       ###")

obtained_words_top = postprocessing.get_top_answers(test_candidates, test_x, output, dictionary, 10)

precision_top = metrics.precision(test_answer, obtained_words_top, STEM_MODE)
recall_top = metrics.recall(test_answer, obtained_words_top, STEM_MODE)
f1_top = metrics.f1(precision_top, recall_top)

print("###    Obtained Scores    ###")
print("### (full dataset, top 10)###")
print("###")
print("### Precision : %.4f" % precision_top)
print("### Recall    : %.4f" % recall_top)
print("### F1        : %.4f" % f1_top)
print("###                       ###")

obtained_words_top = postprocessing.get_top_answers(test_candidates, test_x, output, dictionary, 15)

precision_top = metrics.precision(test_answer, obtained_words_top, STEM_MODE)
recall_top = metrics.recall(test_answer, obtained_words_top, STEM_MODE)
f1_top = metrics.f1(precision_top, recall_top)

print("###    Obtained Scores    ###")
print("### (full dataset, top 15)###")
print("###")
print("### Precision : %.4f" % precision_top)
print("### Recall    : %.4f" % recall_top)
print("### F1        : %.4f" % f1_top)
print("###                       ###")


if DATASET == Semeval2017:
    from eval import anno_generator
    from data.Semeval2017 import eval
    import shutil

    tmp_path = '../data/Semeval2017/tmp/answerrnn2'
    shutil.rmtree(tmp_path, ignore_errors=True)
    anno_generator.write_anno(tmp_path, test_doc_str, obtained_words)
    eval.calculateMeasures("../data/Semeval2017/test", tmp_path, remove_anno=["types"])

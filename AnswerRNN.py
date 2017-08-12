import logging
import os

import numpy as np
from keras import layers
from keras.models import Model, load_model

from data.datasets import Hulth
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
MODEL_PATH = "models/answerrnn.h5"
SHOW_PLOTS = True

# END GLOBAL VARIABLES

# PARAMETERS for networks, tokenizers, etc...

tokenizer = tk.tokenizers.nltk
FILTER = '!"#$%&()*+/:<=>?@[\\]^_`{|}~\t\n'
MAX_DOCUMENT_LENGTH = 550
MAX_ANSWER_LENGTH = 12
MAX_VOCABULARY_SIZE = 20000
EMBEDDINGS_SIZE = 100
BATCH_SIZE = 128
EPOCHS = 1

# END PARAMETERS

logging.info("Loading dataset...")

data = Hulth("data/Hulth2003")

train_doc_str, train_answer_str = data.load_train()
test_doc_str, test_answer_str = data.load_test()
val_doc_str, val_answer_str = data.load_validation()

train_doc, train_answer = tk.tokenize_set(train_doc_str,train_answer_str,tokenizer)
test_doc, test_answer = tk.tokenize_set(test_doc_str,test_answer_str,tokenizer)
val_doc, val_answer = tk.tokenize_set(val_doc_str,val_answer_str,tokenizer)

logging.info("Dataset loaded. Generating candidate keyphrases...")

train_candidates = chunker.extract_candidates_from_set(train_doc_str,tokenizer)
test_candidates = chunker.extract_candidates_from_set(test_doc_str,tokenizer)
val_candidates = chunker.extract_candidates_from_set(val_doc_str,tokenizer)

logging.debug("Candidates recall on training set   : %.4f", metrics.recall(train_answer,train_candidates))
logging.debug("Candidates recall on test set       : %.4f", metrics.recall(test_answer,test_candidates))
logging.debug("Candidates recall on validation set : %.4f", metrics.recall(val_answer,val_candidates))

logging.info("Candidates generated. Preprocessing data...")

train_x,train_y,test_x,test_y,val_x,val_y,embedding_matrix = preprocessing.\
    prepare_answer(train_doc, train_answer, train_candidates,
                   test_doc, test_answer, test_candidates,
                   val_doc,val_answer, val_candidates,
                   max_document_length=MAX_DOCUMENT_LENGTH,
                   max_answer_length=MAX_ANSWER_LENGTH,
                   max_vocabulary_size=MAX_VOCABULARY_SIZE,
                   embeddings_size=EMBEDDINGS_SIZE)

logging.info("Data preprocessing complete.")

if not SAVE_MODEL or not os.path.isfile(MODEL_PATH) :

    logging.debug("Building the network...")
    document = layers.Input(shape=(MAX_DOCUMENT_LENGTH,))
    encoded_document = layers.Embedding(np.shape(embedding_matrix)[0],
                                        EMBEDDINGS_SIZE,
                                        weights=[embedding_matrix],
                                        input_length=MAX_DOCUMENT_LENGTH,
                                        trainable=False)(document)

    encoded_document = layers.Bidirectional(layers.LSTM(int(EMBEDDINGS_SIZE * 2)))\
        (encoded_document)
    encoded_document = layers.Dropout(0.25)(encoded_document)
    encoded_document = layers.Dense(int(EMBEDDINGS_SIZE))\
        (encoded_document)

    candidate = layers.Input(shape=(MAX_ANSWER_LENGTH,))
    encoded_candidate = layers.Embedding(np.shape(embedding_matrix)[0],
                                         EMBEDDINGS_SIZE,
                                         weights=[embedding_matrix],
                                         input_length=MAX_ANSWER_LENGTH,
                                         trainable=False)(candidate)
    encoded_candidate = layers.Bidirectional(layers.LSTM(int(EMBEDDINGS_SIZE)))\
        (encoded_candidate)
    encoded_candidate = layers.Dropout(0.25)(encoded_candidate)
    encoded_candidate = layers.Dense(int(EMBEDDINGS_SIZE))\
        (encoded_candidate)

    merged = layers.add([encoded_document, encoded_candidate])
    prediction = layers.Dense(int(EMBEDDINGS_SIZE / 4))(merged)
    prediction = layers.Dropout(0.25)(prediction)
    prediction = layers.Dense(2, activation='softmax')(prediction)

    model = Model([document, candidate], prediction)

    logging.info("Compiling the network...")
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())

    logging.info("Fitting the network...")
    history = model.fit(train_x, train_y,
                        validation_data=(val_x, val_y),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE)

    if SHOW_PLOTS :
        plots.plot_accuracy(history)
        plots.plot_loss(history)

    if SAVE_MODEL :
        model.save(MODEL_PATH)
        logging.info("Model saved in %s", MODEL_PATH)

else :
    logging.info("Loading existing model from %s...",MODEL_PATH)
    model = load_model(MODEL_PATH)
    logging.info("Completed loading model from file")


logging.info("Predicting on test set...")
output = model.predict(x=test_x, verbose=1)
logging.debug("Shape of output array: %s",np.shape(output))

obtained_tokens = postprocessing.undo_sequential(output)
obtained_words = postprocessing.get_words(test_doc,obtained_tokens)

precision = metrics.precision(test_answer,obtained_words)
recall = metrics.recall(test_answer,obtained_words)
f1 = metrics.f1(precision,recall)

print("###    Obtained Scores    ###")
print("###     (full dataset)    ###")
print("###")
print("### Precision : %.4f" % precision)
print("### Recall    : %.4f" % recall)
print("### F1        : %.4f" % f1)
print("###                       ###")

keras_precision = keras_metrics.keras_precision(test_y,output)
keras_recall = keras_metrics.keras_recall(test_y,output)
keras_f1 = keras_metrics.keras_f1(test_y,output)

print("###    Obtained Scores    ###")
print("###    (fixed dataset)    ###")
print("###")
print("### Precision : %.4f" % keras_precision)
print("### Recall    : %.4f" % keras_recall)
print("### F1        : %.4f" % keras_f1)
print("###                       ###")

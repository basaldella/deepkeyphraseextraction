from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from utils import glove, elmo
from nlp import dictionary as dict
import logging
import numpy as np
import random


def prepare_answer(train_doc, train_answer, train_candidates,
                   test_doc, test_answer, test_candidates,
                   val_doc=None, val_answer=None, val_candidates=None,
                   max_document_length=1000,
                   max_answer_length=20,
                   max_vocabulary_size=50000,
                   embeddings_size=50):
    """
        Prepares a dataset for use by a question-answer like model. This version will use the patterns generated
        previously for the training, test and validation sets as candidate for all three sets.

        :param train_doc: the training documents
        :param train_answer: the KPs for the training documents
        :param train_candidates: the candidate KPs for the training documents
        :param test_doc: the test documents
        :param test_answer: the KPs for the test documents
        :param test_candidates: the candidate KPs for the test documents
        :param val_doc: the validation documents (can be None)
        :param val_answer: the KPs for the validation documents (can be None)
        :param val_candidates: the candidate KPs for the validation documents (can be None)
        :param max_document_length: the maximum length of the documents (shorter documents will be truncated!)
        :param max_answer_length: the maximum length of the answers (shorter answers will be truncated!)
        :param max_vocabulary_size: the maximum size of the vocabulary to use
        (i.e. we keep only the top max_vocabulary_size words)
        :param embeddings_size: the size of the GLoVE embeddings to use
        :return:  a tuple (train_x, train_y, test_x, test_y, val_x, val_y, embedding_matrix) containing the training,
        test and validation set, and an embedding matrix for an Embedding layer
        """

    # Prepare validation return data
    val_q = None
    val_a = None
    val_y = None

    # Prepare the return values: lists that will hold questions (documents), answers (keyphrases), and truth values
    train_q = []
    test_q = []
    train_a = []
    test_a = []
    train_y = []
    test_y = []

    if val_doc and val_answer:
        val_q = []
        val_a = []
        val_y = []

    documents_full = []
    for key, doc in train_doc.items():
        documents_full.append(token for token in doc)
    for key, doc in test_doc.items():
        documents_full.append(token for token in doc)

    if val_doc and val_answer:
        for key, doc in val_doc.items():
            documents_full.append(token for token in doc)

    logging.debug("Fitting dictionary on %s documents..." % len(documents_full))

    dictionary = dict.Dictionary(num_words=max_vocabulary_size)
    dictionary.fit_on_texts(documents_full)

    logging.debug("Dictionary fitting completed. Found %s unique tokens" % len(dictionary.word_index))

    # Pair up each document with a candidate keyphrase and its truth value
    for key, document in train_doc.items():
        doc_sequence = dictionary.token_list_to_sequence(document)
        for kp in train_candidates[key]:
            train_q.append(doc_sequence)
            train_a.append(dictionary.token_list_to_sequence(kp))
            train_y.append([0, 1] if kp in train_answer[key] else [1, 0])

    for key, document in test_doc.items():
        doc_sequence = dictionary.token_list_to_sequence(document)
        for kp in test_candidates[key]:
            test_q.append(doc_sequence)
            test_a.append(dictionary.token_list_to_sequence(kp))
            test_y.append([0, 1] if kp in test_answer[key] else [1, 0])

    if val_doc and val_answer:
        for key, document in val_doc.items():
            doc_sequence = dictionary.token_list_to_sequence(document)
            for kp in val_candidates[key]:
                val_q.append(doc_sequence)
                val_a.append(dictionary.token_list_to_sequence(kp))
                val_y.append([0, 1] if kp in val_answer[key] else [1, 0])

    logging.debug("Longest training document   : %s tokens" % len(max(train_q, key=len)))
    logging.debug("Longest training answer     : %s tokens" % len(max(train_a, key=len)))
    logging.debug("Longest test document       : %s tokens" % len(max(test_q, key=len)))
    logging.debug("Longest test answer         : %s tokens" % len(max(test_a, key=len)))
    if val_doc and val_answer:
        logging.debug("Longest validation document : %s tokens" % len(max(val_q, key=len)))
        logging.debug("Longest validation answer   : %s tokens" % len(max(val_a, key=len)))

    train_q = np.asarray(pad_sequences(train_q, maxlen=max_document_length, padding='post', truncating='post'))
    train_a = np.asarray(pad_sequences(train_a, maxlen=max_answer_length, padding='post', truncating='post'))

    test_q = np.asarray(pad_sequences(test_q, maxlen=max_document_length, padding='post', truncating='post'))
    test_a = np.asarray(pad_sequences(test_a, maxlen=max_answer_length, padding='post', truncating='post'))

    if val_doc and val_answer:
        val_q = np.asarray(pad_sequences(val_q, maxlen=max_document_length, padding='post', truncating='post'))
        val_a = np.asarray(pad_sequences(val_a, maxlen=max_answer_length, padding='post', truncating='post'))

    logging.debug("Training set documents size   : %s", np.shape(train_q))
    logging.debug("Training set answers size     : %s", np.shape(train_a))
    logging.debug("Test set documents size       : %s", np.shape(test_q))
    logging.debug("Test set answers size         : %s ", np.shape(test_a))

    if val_doc and val_answer:
        logging.debug("Validation set documents size : %s", np.shape(val_q))
        logging.debug("Validation set answers size   : %s ", np.shape(val_a))

    # prepare the matrix for the embedding layer
    word_index = dictionary.word_index

    num_words = min(max_vocabulary_size, 1 + len(word_index))

    logging.debug("Building embedding matrix of size [%s,%s]..." % (num_words, embeddings_size))

    embedding_matrix = np.zeros((num_words, embeddings_size))

    if embeddings_size <= 300:
        embeddings_index = glove.load_glove('', embeddings_size)
        for word, i in word_index.items():
            if i >= num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    else:
        print('elmo')
        embedding_vectors = elmo.load_elmo(word_index)
        embedding_matrix[1:] = embedding_vectors

    return [train_q, train_a], train_y, [test_q, test_a], test_y, [val_q, val_a], val_y, embedding_matrix, dictionary


def prepare_answer_2(train_doc, train_answer, train_candidates,
                     test_doc, test_answer, test_candidates,
                     val_doc=None, val_answer=None, val_candidates=None,
                     max_document_length=1000,
                     max_answer_length=20,
                     max_vocabulary_size=50000,
                     embeddings_size=50):
    """
        Prepares a dataset for use by a question-answer like model. This version will use the patterns generated
        previously for the test and validation sets as candidate for these sets, and mix the correct answers with
        wrong patterns on the training set to build in order to have balanced data for training.

        :param train_doc: the training documents
        :param train_answer: the KPs for the training documents
        :param train_candidates: the candidate KPs for the training documents
        :param test_doc: the test documents
        :param test_answer: the KPs for the test documents
        :param test_candidates: the candidate KPs for the test documents
        :param val_doc: the validation documents (can be None)
        :param val_answer: the KPs for the validation documents (can be None)
        :param val_candidates: the candidate KPs for the validation documents (can be None)
        :param max_document_length: the maximum length of the documents (shorter documents will be truncated!)
        :param max_answer_length: the maximum length of the answers (shorter answers will be truncated!)
        :param max_vocabulary_size: the maximum size of the vocabulary to use
        (i.e. we keep only the top max_vocabulary_size words)
        :param embeddings_size: the size of the GLoVE embeddings to use
        :return:  a tuple (train_x, train_y, test_x, test_y, val_x, val_y, embedding_matrix) containing the training,
        test and validation set, and an embedding matrix for an Embedding layer
        """

    # Prepare validation return data
    val_q = None
    val_a = None
    val_y = None

    val_q_balanced = None
    val_a_balanced = None
    val_y_balanced = None

    # Prepare the return values: lists that will hold questions (documents), answers (keyphrases), and truth values
    train_q = []
    test_q = []
    train_a = []
    test_a = []
    train_y = []
    test_y = []

    if val_doc and val_answer:
        val_q = []
        val_a = []
        val_y = []
        val_q_balanced = []
        val_a_balanced = []
        val_y_balanced = []

    documents_full = []
    for key, doc in train_doc.items():
        documents_full.append(token for token in doc)
    for key, doc in test_doc.items():
        documents_full.append(token for token in doc)

    if val_doc and val_answer:
        for key, doc in val_doc.items():
            documents_full.append(token for token in doc)

    logging.debug("Fitting dictionary on %s documents..." % len(documents_full))

    dictionary = dict.Dictionary(num_words=max_vocabulary_size)
    dictionary.fit_on_texts(documents_full)

    logging.debug("Dictionary fitting completed. Found %s unique tokens" % len(dictionary.word_index))

    # Pair up each document with a candidate keyphrase and its truth value
    for key, document in train_doc.items():
        doc_sequence = dictionary.token_list_to_sequence(document)

        # select wrong candidates (possibly, in same quantity as good answers)
        wrong_candidates = list(train_candidates[key])
        for answer in train_answer[key]:
            if answer in wrong_candidates:
                wrong_candidates.remove(answer)

        while len(wrong_candidates) > len(train_answer[key]):
            random_candidate = random.choice(wrong_candidates)
            wrong_candidates.remove(random_candidate)

        # append wrong candidates
        for kp in wrong_candidates:
            train_q.append(doc_sequence)
            train_a.append(dictionary.token_list_to_sequence(kp))
            train_y.append([1, 0])

        # append true answers
        for kp in train_answer[key]:
            train_q.append(doc_sequence)
            train_a.append(dictionary.token_list_to_sequence(kp))
            train_y.append([0, 1])

    if val_doc and val_answer:
        for key, document in val_doc.items():
            doc_sequence = dictionary.token_list_to_sequence(document)

            # select wrong candidates (possibly, in same quantity as good answers)
            wrong_candidates = list(val_candidates[key])
            for answer in val_answer[key]:
                if answer in wrong_candidates:
                    wrong_candidates.remove(answer)

            while len(wrong_candidates) > len(val_answer[key]):
                random_candidate = random.choice(wrong_candidates)
                wrong_candidates.remove(random_candidate)

            # append wrong candidates
            for kp in wrong_candidates:
                val_q_balanced.append(doc_sequence)
                val_a_balanced.append(dictionary.token_list_to_sequence(kp))
                val_y_balanced.append([1, 0])

            # append true answers
            for kp in val_answer[key]:
                val_q_balanced.append(doc_sequence)
                val_a_balanced.append(dictionary.token_list_to_sequence(kp))
                val_y_balanced.append([0, 1])

    # for the other sets, just pick the auto-generated candidates
    for key, document in test_doc.items():
        doc_sequence = dictionary.token_list_to_sequence(document)
        for kp in test_candidates[key]:
            test_q.append(doc_sequence)
            test_a.append(dictionary.token_list_to_sequence(kp))
            test_y.append([0, 1] if kp in test_answer[key] else [1, 0])

    if val_doc and val_answer:
        for key, document in val_doc.items():
            doc_sequence = dictionary.token_list_to_sequence(document)
            for kp in val_candidates[key]:
                val_q.append(doc_sequence)
                val_a.append(dictionary.token_list_to_sequence(kp))
                val_y.append([0, 1] if kp in val_answer[key] else [1, 0])

    logging.debug("Longest training document            : %s tokens" % len(max(train_q, key=len)))
    logging.debug("Longest training answer              : %s tokens" % len(max(train_a, key=len)))
    logging.debug("Longest test document                : %s tokens" % len(max(test_q, key=len)))
    logging.debug("Longest test answer                  : %s tokens" % len(max(test_a, key=len)))
    if val_doc and val_answer:
        logging.debug("Longest validation document          : %s tokens" % len(max(val_q, key=len)))
        logging.debug("Longest validation answer            : %s tokens" % len(max(val_a, key=len)))
        logging.debug("Longest balanced validation document : %s tokens" % len(max(val_q, key=len)))
        logging.debug("Longest balanced validation answer   : %s tokens" % len(max(val_a, key=len)))

    train_q = np.asarray(pad_sequences(train_q, maxlen=max_document_length, padding='post', truncating='post'))
    train_a = np.asarray(pad_sequences(train_a, maxlen=max_answer_length, padding='post', truncating='post'))

    test_q = np.asarray(pad_sequences(test_q, maxlen=max_document_length, padding='post', truncating='post'))
    test_a = np.asarray(pad_sequences(test_a, maxlen=max_answer_length, padding='post', truncating='post'))

    if val_doc and val_answer:
        val_q = np.asarray(pad_sequences(val_q, maxlen=max_document_length, padding='post', truncating='post'))
        val_a = np.asarray(pad_sequences(val_a, maxlen=max_answer_length, padding='post', truncating='post'))
        val_q_balanced = np.asarray(pad_sequences(val_q_balanced, maxlen=max_document_length, padding='post', truncating='post'))
        val_a_balanced = np.asarray(pad_sequences(val_a_balanced, maxlen=max_answer_length, padding='post', truncating='post'))

    logging.debug("Training set documents size            : %s", np.shape(train_q))
    logging.debug("Training set answers size              : %s", np.shape(train_a))
    logging.debug("Test set documents size                : %s", np.shape(test_q))
    logging.debug("Test set answers size                  : %s ", np.shape(test_a))

    if val_doc and val_answer:
        logging.debug("Validation set documents size          : %s", np.shape(val_q))
        logging.debug("Validation set answers size            : %s ", np.shape(val_a))
        logging.debug("Balanced Validation set documents size : %s", np.shape(val_q_balanced))
        logging.debug("Balanced Validation set answers size   : %s ", np.shape(val_a_balanced))

    # prepare the matrix for the embedding layer
    word_index = dictionary.word_index

    num_words = min(max_vocabulary_size, 1 + len(word_index))

    logging.debug("Building embedding matrix of size [%s,%s]..." % (num_words, embeddings_size))

    embedding_matrix = np.zeros((num_words, embeddings_size))

    if embeddings_size <= 300:
        embeddings_index = glove.load_glove('../data/glove', embeddings_size)
        for word, i in word_index.items():
            if i >= num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    else:
        print('elmo')
        embedding_vectors = elmo.load_elmo(word_index)
        embedding_matrix[1:] = embedding_vectors

    return [train_q, train_a], train_y, [test_q, test_a], test_y, [val_q, val_a], val_y, \
           [val_q_balanced, val_a_balanced], val_y_balanced, embedding_matrix, dictionary


def prepare_sequential(train_doc, train_answer, test_doc, test_answer, val_doc, val_answer,
                       max_document_length=1000,
                       max_vocabulary_size=50000,
                       embeddings_size=50,
                       stem_test = False):
    """
        Prepares a dataset for use by a sequential, categorical model.

        :param train_doc: the training documents
        :param train_answer: the KPs for the training documents
        :param test_doc: the test documents
        :param test_answer: the KPs for the test documents
        :param val_doc: the validation documents (can be None)
        :param val_answer: the KPs for the validation documents (can be None)
        :param max_document_length: the maximum length of the documents (shorter documents will be truncated!)
        :param max_vocabulary_size: the maximum size of the vocabulary to use
        (i.e. we keep only the top max_vocabulary_size words)
        :param embeddings_size: the size of the GLoVE embeddings to use
        :param stem_test: set the value to True if the test set answers are stemmed
        :return: a tuple (train_x, train_y, test_x, test_y, val_x, val_y, embedding_matrix) containing the training,
        test and validation set, and an embedding matrix for an Embedding layer
        """

    train_answer_seq = make_sequential(train_doc, train_answer)

    if not stem_test:
        test_answer_seq = make_sequential(test_doc, test_answer)
    else:
        import copy
        stemmed_test_doc = copy.deepcopy(test_doc)
        stemmed_test_doc = stem_dataset(stemmed_test_doc)
        test_answer_seq = make_sequential(stemmed_test_doc,test_answer)

    # Prepare validation return data
    val_x = None
    val_y = None

    if val_doc and val_answer:
        val_answer_seq = make_sequential(val_doc, val_answer)

    # Transform the documents to sequence
    documents_full = []
    train_y = []
    test_y = []

    if val_doc and val_answer:
        val_y = []

    for key, doc in train_doc.items():
        documents_full.append(token for token in doc)
        train_y.append(train_answer_seq[key])
    for key, doc in test_doc.items():
        documents_full.append(token for token in doc)
        test_y.append(test_answer_seq[key])

    if val_doc and val_answer:
        for key, doc in val_doc.items():
            documents_full.append(token for token in doc)
            val_y.append(val_answer_seq[key])

    logging.debug("Fitting dictionary on %s documents..." % len(documents_full))

    dictionary = dict.Dictionary(num_words=max_vocabulary_size)
    dictionary.fit_on_texts(documents_full)

    logging.debug("Dictionary fitting completed. Found %s unique tokens" % len(dictionary.word_index))

    # Now we can prepare the actual input
    train_x = dictionary.texts_to_sequences(train_doc.values())
    test_x = dictionary.texts_to_sequences(test_doc.values())
    if val_doc and val_answer:
        val_x = dictionary.texts_to_sequences(val_doc.values())

    logging.debug("Longest training document : %s tokens" % len(max(train_x, key=len)))
    logging.debug("Longest test document :     %s tokens" % len(max(test_x, key=len)))
    if val_doc and val_answer:
        logging.debug("Longest validation document : %s tokens" % len(max(val_x, key=len)))

    train_x = np.asarray(pad_sequences(train_x, maxlen=max_document_length, padding='post', truncating='post'))
    train_y = pad_sequences(train_y, maxlen=max_document_length, padding='post', truncating='post')
    train_y = make_categorical(train_y)

    test_x = np.asarray(pad_sequences(test_x, maxlen=max_document_length, padding='post', truncating='post'))
    test_y = pad_sequences(test_y, maxlen=max_document_length, padding='post', truncating='post')
    test_y = make_categorical(test_y)

    if val_doc and val_answer:
        val_x = np.asarray(pad_sequences(val_x, maxlen=max_document_length, padding='post', truncating='post'))
        val_y = pad_sequences(val_y, maxlen=max_document_length, padding='post', truncating='post')
        val_y = make_categorical(val_y)

    logging.debug("Training set samples size   : %s", np.shape(train_x))
    logging.debug("Training set answers size   : %s", np.shape(train_y))
    logging.debug("Test set samples size       : %s", np.shape(test_x))
    logging.debug("Test set answers size       : %s ", np.shape(test_y))

    if val_doc and val_answer:
        logging.debug("Validation set samples size : %s", np.shape(val_x))
        logging.debug("Validation set answers size : %s ", np.shape(val_y))

    # prepare the matrix for the embedding layer
    word_index = dictionary.word_index

    num_words = min(max_vocabulary_size, 1 + len(word_index))

    logging.debug("Building embedding matrix of size [%s,%s]..." % (num_words, embeddings_size))

    embedding_matrix = np.zeros((num_words, embeddings_size))

    if embeddings_size <= 300:
        embeddings_index = glove.load_glove('../data/glove', embeddings_size)
        for word, i in word_index.items():
            if i >= num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    else:
        print('elmo')
        embedding_vectors = elmo.load_elmo(word_index)
        embedding_matrix[1:] = embedding_vectors

    return train_x, train_y, test_x, test_y, val_x, val_y, embedding_matrix


def prepare_sequential_elmo3(train_doc, train_answer, test_doc, test_answer, val_doc, val_answer,
                             max_document_length=1000,
                             max_vocabulary_size=50000,
                             stem_test=False):
    """
        Prepares a dataset for use by a sequential, categorical model.

        :param train_doc: the training documents
        :param train_answer: the KPs for the training documents
        :param test_doc: the test documents
        :param test_answer: the KPs for the test documents
        :param val_doc: the validation documents (can be None)
        :param val_answer: the KPs for the validation documents (can be None)
        :param max_document_length: the maximum length of the documents (shorter documents will be truncated!)
        :param max_vocabulary_size: the maximum size of the vocabulary to use
        (i.e. we keep only the top max_vocabulary_size words)
        :param stem_test: set the value to True if the test set answers are stemmed
        :return: a tuple (train_x, train_y, test_x, test_y, val_x, val_y, embedding_matrix) containing the training,
        test and validation set, and an embedding matrix for an Embedding layer
        """

    train_answer_seq = make_sequential(train_doc, train_answer)

    if not stem_test:
        test_answer_seq = make_sequential(test_doc, test_answer)
    else:
        import copy
        stemmed_test_doc = copy.deepcopy(test_doc)
        stemmed_test_doc = stem_dataset(stemmed_test_doc)
        test_answer_seq = make_sequential(stemmed_test_doc, test_answer)

    # Prepare validation return data
    val_x = None
    val_y = None

    if val_doc and val_answer:
        val_answer_seq = make_sequential(val_doc, val_answer)

    # Transform the documents to sequence
    documents_full = []
    train_y = []
    test_y = []

    if val_doc and val_answer:
        val_y = []

    for key, doc in train_doc.items():
        documents_full.append(token for token in doc)
        train_y.append(train_answer_seq[key])
    for key, doc in test_doc.items():
        documents_full.append(token for token in doc)
        test_y.append(test_answer_seq[key])

    if val_doc and val_answer:
        for key, doc in val_doc.items():
            documents_full.append(token for token in doc)
            val_y.append(val_answer_seq[key])

    logging.debug("Fitting dictionary on %s documents..." % len(documents_full))

    dictionary = dict.Dictionary(num_words=max_vocabulary_size)
    dictionary.fit_on_texts(documents_full)

    logging.debug("Dictionary fitting completed. Found %s unique tokens" % len(dictionary.word_index))

    # Now we can prepare the actual input
    train_x = dictionary.texts_to_sequences(train_doc.values())
    test_x = dictionary.texts_to_sequences(test_doc.values())
    if val_doc and val_answer:
        val_x = dictionary.texts_to_sequences(val_doc.values())

    logging.debug("Longest training document : %s tokens" % len(max(train_x, key=len)))
    logging.debug("Longest test document :     %s tokens" % len(max(test_x, key=len)))
    if val_doc and val_answer:
        logging.debug("Longest validation document : %s tokens" % len(max(val_x, key=len)))

    train_x = np.asarray(pad_sequences(train_x, maxlen=max_document_length, padding='post', truncating='post'))
    train_y = pad_sequences(train_y, maxlen=max_document_length, padding='post', truncating='post')
    train_y = make_categorical(train_y)

    test_x = np.asarray(pad_sequences(test_x, maxlen=max_document_length, padding='post', truncating='post'))
    test_y = pad_sequences(test_y, maxlen=max_document_length, padding='post', truncating='post')
    test_y = make_categorical(test_y)

    if val_doc and val_answer:
        val_x = np.asarray(pad_sequences(val_x, maxlen=max_document_length, padding='post', truncating='post'))
        val_y = pad_sequences(val_y, maxlen=max_document_length, padding='post', truncating='post')
        val_y = make_categorical(val_y)

    logging.debug("Training set samples size   : %s", np.shape(train_x))
    logging.debug("Training set answers size   : %s", np.shape(train_y))
    logging.debug("Test set samples size       : %s", np.shape(test_x))
    logging.debug("Test set answers size       : %s ", np.shape(test_y))

    if val_doc and val_answer:
        logging.debug("Validation set samples size : %s", np.shape(val_x))
        logging.debug("Validation set answers size : %s ", np.shape(val_y))

    # prepare the matrix for the embedding layer
    word2index = dictionary.word_index

    return train_x, train_y, test_x, test_y, val_x, val_y, word2index


def prepare_sequential_sent(train_doc, train_answer, test_doc, test_answer, val_doc, val_answer,
                            max_document_length=1000,
                            max_sentences_in_doc=30,
                            max_tokens_in_sentence=200,
                            max_vocabulary_size=50000,
                            stem_test=False):
    """
        Prepares a dataset for use by a sequential, categorical model.

        :param train_doc: the training documents
        :param train_answer: the KPs for the training documents
        :param test_doc: the test documents
        :param test_answer: the KPs for the test documents
        :param val_doc: the validation documents (can be None)
        :param val_answer: the KPs for the validation documents (can be None)
        :param max_document_length: the maximum length of the documents (shorter documents will be truncated!)
        :param max_vocabulary_size: the maximum size of the vocabulary to use
        (i.e. we keep only the top max_vocabulary_size words)
        :param stem_test: set the value to True if the test set answers are stemmed
        :return: a tuple (train_x, train_y, test_x, test_y, val_x, val_y, embedding_matrix) containing the training,
        test and validation set, and an embedding matrix for an Embedding layer
        """

    flat_train_doc = make_flat_documents(train_doc)
    train_answer_seq = make_sequential(flat_train_doc, train_answer)

    flat_test_doc = make_flat_documents(test_doc)
    if not stem_test:
        test_answer_seq = make_sequential(flat_test_doc, test_answer)
    else:
        import copy
        stemmed_test_doc = copy.deepcopy(flat_test_doc)
        stemmed_test_doc = stem_dataset(stemmed_test_doc)
        test_answer_seq = make_sequential(stemmed_test_doc, test_answer)

    # Prepare validation return data
    val_x = None
    val_y = None

    flat_val_doc = make_flat_documents(val_doc)
    if flat_val_doc and val_answer:
        val_answer_seq = make_sequential(flat_val_doc, val_answer)

    # Transform the documents to sequence
    documents_full = []
    train_y = []
    test_y = []

    if flat_val_doc and val_answer:
        val_y = []

    for key, doc in flat_train_doc.items():
        documents_full.append(token for token in doc)
        train_y.append(train_answer_seq[key])
    for key, doc in flat_test_doc.items():
        documents_full.append(token for token in doc)
        test_y.append(test_answer_seq[key])

    if flat_val_doc and val_answer:
        for key, doc in flat_val_doc.items():
            documents_full.append(token for token in doc)
            val_y.append(val_answer_seq[key])

    logging.debug("Fitting dictionary on %s documents..." % len(documents_full))

    dictionary = dict.Dictionary(num_words=max_vocabulary_size)
    dictionary.fit_on_texts(documents_full)

    logging.debug("Dictionary fitting completed. Found %s unique tokens" % len(dictionary.word_index))

    # Now we can prepare the actual input
    '''
    # train_x = dictionary.texts_to_sequences(train_doc.values())
    train_x = list()
    for docs in train_doc.values():
        sentences = dictionary.texts_to_sequences(docs)
        train_x.append(sentences)

    # test_x = dictionary.texts_to_sequences(test_doc.values())
    test_x = list()
    for docs in test_doc.values():
        sentences = dictionary.texts_to_sequences(docs)
        test_x.append(sentences)

    if val_doc and val_answer:
        # val_x = dictionary.texts_to_sequences(val_doc.values())
        val_x = list()
        for docs in val_doc.values():
            sentences = dictionary.texts_to_sequences(docs)
            val_x.append(sentences)
    '''
    # all tensors '<name>_x' of shape: (batch, max_sentences_in_doc, max_tokens_in_sentence)
    empty_sentence = [0]*max_tokens_in_sentence
    # train_x = dictionary.texts_to_sequences(train_doc.values())
    train_x = list()
    for docs in train_doc.values():
        sentences = dictionary.texts_to_sequences(docs)
        for i in range(len(sentences), max_sentences_in_doc):
            sentences.append(empty_sentence)
        train_x.append(sentences)

    # test_x = dictionary.texts_to_sequences(test_doc.values())
    test_x = list()
    for docs in test_doc.values():
        sentences = dictionary.texts_to_sequences(docs)
        for i in range(len(sentences), max_sentences_in_doc):
            sentences.append(empty_sentence)
        test_x.append(sentences)

    if val_doc and val_answer:
        # val_x = dictionary.texts_to_sequences(val_doc.values())
        val_x = list()
        for docs in val_doc.values():
            sentences = dictionary.texts_to_sequences(docs)
            for i in range(len(sentences), max_sentences_in_doc):
                sentences.append(empty_sentence)
            val_x.append(sentences)

    logging.debug("Longest training document : %s tokens" % len(max(train_x, key=len)))
    logging.debug("Longest test document :     %s tokens" % len(max(test_x, key=len)))
    if val_doc and val_answer:
        logging.debug("Longest validation document : %s tokens" % len(max(val_x, key=len)))

    # train_x = np.asarray(pad_sequences(train_x, maxlen=max_document_length, padding='post', truncating='post'))
    for i, docs in enumerate(train_x):
        train_x[i] = pad_sequences(docs, maxlen=max_tokens_in_sentence, padding='post', truncating='post')
    train_x = np.asarray(train_x)
    train_y = pad_sequences(train_y, maxlen=max_document_length, padding='post', truncating='post')
    train_y = make_categorical(train_y)

    # test_x = np.asarray(pad_sequences(test_x, maxlen=max_document_length, padding='post', truncating='post'))
    for i, docs in enumerate(test_x):
        test_x[i] = pad_sequences(docs, maxlen=max_tokens_in_sentence, padding='post', truncating='post')
    test_x = np.asarray(test_x)
    test_y = pad_sequences(test_y, maxlen=max_document_length, padding='post', truncating='post')
    test_y = make_categorical(test_y)

    if val_doc and val_answer:
        # val_x = np.asarray(pad_sequences(val_x, maxlen=max_document_length, padding='post', truncating='post'))
        for i, docs in enumerate(val_x):
            val_x[i] = pad_sequences(docs, maxlen=max_tokens_in_sentence, padding='post', truncating='post')
        val_x = np.asarray(val_x)
        val_y = pad_sequences(val_y, maxlen=max_document_length, padding='post', truncating='post')
        val_y = make_categorical(val_y)

    logging.debug("Training set samples size   : %s", np.shape(train_x))
    logging.debug("Training set answers size   : %s", np.shape(train_y))
    logging.debug("Test set samples size       : %s", np.shape(test_x))
    logging.debug("Test set answers size       : %s ", np.shape(test_y))

    if val_doc and val_answer:
        logging.debug("Validation set samples size : %s", np.shape(val_x))
        logging.debug("Validation set answers size : %s ", np.shape(val_y))

    # prepare the matrix for the embedding layer
    word2index = dictionary.word_index

    return train_x, train_y, test_x, test_y, val_x, val_y, word2index


def prepare_sequential_elmo(train_doc, train_answer, test_doc, test_answer, val_doc, val_answer,
                            max_document_length=1000,
                            stem_test=False):
    """
        Prepares a dataset for use by a sequential, categorical model.

        :param train_doc: the training documents
        :param train_answer: the KPs for the training documents
        :param test_doc: the test documents
        :param test_answer: the KPs for the test documents
        :param val_doc: the validation documents (can be None)
        :param val_answer: the KPs for the validation documents (can be None)
        :param max_document_length: the maximum length of the documents (shorter documents will be truncated!)
        (i.e. we keep only the top max_vocabulary_size words)
        :param stem_test: set the value to True if the test set answers are stemmed
        :return: a tuple (train_x_e, train_y, test_x_e, test_y, val_x_e, val_y) containing the training,
        test and validation set
        """

    train_answer_seq = make_sequential(train_doc, train_answer)

    if not stem_test:
        test_answer_seq = make_sequential(test_doc, test_answer)
    else:
        import copy
        stemmed_test_doc = copy.deepcopy(test_doc)
        stemmed_test_doc = stem_dataset(stemmed_test_doc)
        test_answer_seq = make_sequential(stemmed_test_doc,test_answer)

    # Prepare validation return data
    # val_x = None
    val_y = None

    if val_doc and val_answer:
        val_answer_seq = make_sequential(val_doc, val_answer)

    # Transform the documents to sequence
    documents_full = []
    train_y = []
    test_y = []

    if val_doc and val_answer:
        val_y = []

    for key, doc in train_doc.items():
        documents_full.append(token for token in doc)
        train_y.append(train_answer_seq[key])
    for key, doc in test_doc.items():
        documents_full.append(token for token in doc)
        test_y.append(test_answer_seq[key])

    if val_doc and val_answer:
        for key, doc in val_doc.items():
            documents_full.append(token for token in doc)
            val_y.append(val_answer_seq[key])

    # train_x = np.asarray(pad_sequences(train_x, maxlen=max_document_length, padding='post', truncating='post'))
    train_x_e = list(train_doc.values())
    train_x_e = np.asarray(pad_sequences(train_x_e,
                                         maxlen=max_document_length,
                                         dtype='object',
                                         padding='post',
                                         truncating='post'))
    train_x_e = train_x_e.astype(str)
    train_x_e = np.where(train_x_e == '0.0', '', train_x_e)
    train_y = pad_sequences(train_y, maxlen=max_document_length, padding='post', truncating='post')
    train_y = make_categorical(train_y)

    # test_x = np.asarray(pad_sequences(test_x, maxlen=max_document_length, padding='post', truncating='post'))
    test_x_e = list(test_doc.values())
    test_x_e = np.asarray(pad_sequences(test_x_e,
                                        maxlen=max_document_length,
                                        dtype='object',
                                        padding='post',
                                        truncating='post'))
    test_x_e = test_x_e.astype(str)
    test_x_e = np.where(test_x_e == '0.0', '', test_x_e)
    test_y = pad_sequences(test_y, maxlen=max_document_length, padding='post', truncating='post')
    test_y = make_categorical(test_y)

    if val_doc and val_answer:
        # val_x = np.asarray(pad_sequences(val_x, maxlen=max_document_length, padding='post', truncating='post'))
        val_x_e = list(val_doc.values())
        val_x_e = np.asarray(pad_sequences(val_x_e,
                                           maxlen=max_document_length,
                                           dtype='object',
                                           padding='post',
                                           truncating='post'))
        val_x_e = val_x_e.astype(str)
        val_x_e = np.where(val_x_e == '0.0', '', val_x_e)
        val_y = pad_sequences(val_y, maxlen=max_document_length, padding='post', truncating='post')
        val_y = make_categorical(val_y)

    logging.debug("Longest training document : %s tokens" % len(max(train_x_e, key=len)))
    logging.debug("Longest test document :     %s tokens" % len(max(test_x_e, key=len)))
    if val_doc and val_answer:
        logging.debug("Longest validation document : %s tokens" % len(max(val_x_e, key=len)))

    logging.debug("Training set samples size   : %s", np.shape(train_x_e))
    logging.debug("Training set answers size   : %s", np.shape(train_y))
    logging.debug("Test set samples size       : %s", np.shape(test_x_e))
    logging.debug("Test set answers size       : %s ", np.shape(test_y))

    if val_doc and val_answer:
        logging.debug("Validation set samples size : %s", np.shape(val_x_e))
        logging.debug("Validation set answers size : %s ", np.shape(val_y))

    return train_x_e, train_y, test_x_e, test_y, val_x_e, val_y


def make_sequential(documents, answers):
    """
    Transform an answer-based dataset (i.e. with a list of
    documents and a list of keyphrases) to a sequential, ner-like
    dataset, i.e. where the answer set for each document is composed
    by the lists of the documents' tokens marked as non-keyphrase (0),
    beginning of keyphrase (1) and inside-keyphrase (2).

    For example, for the tokens

    "I am a python developer since today."

    If the keyphrases are "python developer" and "today"" the answer
    set for these tokens is

    "[0 0 0 1 2 0 1]"

    :param documents: the list of documents
    :param answers: the list of keyphrases
    :return: the new answer set
    """

    seq_answers = {}

    for key, document in documents.items():
        doc_answers_set = answers[key]
        # Sort by length of the keyphrase. We process shorter KPs first so
        # that if they are contained by a longer KP we'll simply overwrite
        # the shorter one with the longer one
        doc_answers_set.sort(key=lambda a: len(a))

        # This field will contain the answer.
        # We initialize it as a list of zeros and we will fill it
        # with 1s and 2s later
        doc_answers_seq = [0] * len(document)

        for answer in doc_answers_set:

            if not answer:
                continue

            # Find where the first word the KP appears
            # print('doc_answers_set: ' + str(doc_answers_set))
            # print('len(answer): ' + str(len(answer)) + '; key: ' + key)
            # print('answer[0]: ' + str(answer[0]))
            appearances = [i for i, word in enumerate(document) if word == answer[0]]
            for idx in appearances:
                is_kp = True
                # Check if the KP matches also from its second word on
                for i in range(1, len(answer)):

                    if (i + idx) < len(document):
                        is_kp = answer[i] == document[i + idx]
                    else:
                        # We reached the end of the document
                        is_kp = False

                # If we found an actual KP, mark the tokens in the output list.
                if is_kp:
                    doc_answers_seq[idx] = 1
                    for i in range(1, len(answer)):
                        doc_answers_seq[idx + i] = 2

                        # for
        # for

        seq_answers[key] = doc_answers_seq

    return seq_answers


def make_categorical(x):
    """
    Transform a two-dimensional list into a 3-dimensional array. The 2nd
    dimension of the input list becomes a one-hot 2D array, e.g.
    if the input is [[1,2,0],...], the output will be
    [[[0,1,0],[0,0,1],[1,0,0]],...]

    :param x: a 2D-list
    :return: a 3D-numpy array
    """

    # How many categories do we have?
    num_categories = max([item for sublist in x for item in sublist]) + 1

    # Prepare numpy output
    new_x = np.zeros((len(x), len(x[0]), num_categories))

    # Use keras to make actual categorical transformation
    i = 0
    for doc in x:
        new_doc = np_utils.to_categorical(doc, num_classes=num_categories)
        new_x[i] = new_doc
        i += 1

    return new_x


def stem_dataset(dataset):

    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()

    for key, tokens in dataset.items():
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        dataset[key] = stemmed_tokens

    return dataset


def words_in_documents(documents, answers):

    found_tokens = 0
    not_found_tokens = 0
    tokens_in_documents = 0
    for doc_id, answer in answers.items():
        tokens_in_documents += len(documents[doc_id])
        for kp in answer:
            for token in kp:
                if token in documents[doc_id]:
                    found_tokens += 1
                else:
                    not_found_tokens += 1

    # return (found_tokens * 1.0) / tokens_in_documents
    return found_tokens, not_found_tokens, tokens_in_documents


def make_flat_documents(documents):
    """
    Transform the dictionary of input documents splitted in sentences and then tokenized (dict of lists of lists)
    in a dictionary of flat documents with tokenized texts (dict of lists)
    For example, typical input documents are in the form:
    ['doc_id': [[word11, word12, ..., word1n], [word21, word22, ..., word2m], ..., [wordk1, wordk2, ..., wordkl]]]
    and the output:
    ['doc_id: [word11, word12, ..., wordkl]]

    :param documents: the dictionary of input documents splitted in sentences and tokeinized
    :return: the dictionary of input documents with tokenized text
    """

    flat_documents = {}

    for key, document in documents.items():
        flat_doc = [item for sublist in document for item in sublist]  # pk it works!
        flat_documents[key] = flat_doc

    return flat_documents

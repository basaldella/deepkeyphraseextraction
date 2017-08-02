from utils import tokenizer as tk
import logging
import os


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
            # Find where the first word the KP appears
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


class Dataset(object):
    """
    An abstract class that represents a dataset.
    """

    def __init__(self, name, path):
        self.path = path
        self.name = name
        self.test_documents = None
        self.test_answers = None
        self.train_documents = None
        self.train_answers = None

        logging.info("Initialized dataset %s from folder %s" %
                     (self.name, self.path))

    def __str__(self):
        return 'Dataset %s located in folder %s' % (self.name, self.path)

    def _load_test_documents(self):
        """
        Loads the test documents.

        :return: a list of documents.
        """
        raise NotImplementedError

    def _load_test_answers(self):
        """
        Loads the answers for the test documents.
        :return: a list of answers.
        """
        raise NotImplementedError

    def _load_train_documents(self):
        """
        Loads the train documents.

        :return: a list of documents.
        """
        raise NotImplementedError

    def _load_train_answers(self):
        """
        Loads the answers for the train documents.
        :return: a list of answers.
        """
        raise NotImplementedError

    def load_test(self):
        """
        Loads the test documents and their answers.
        :return: a tuple containing the test documents and the test answers.
        """

        if not self.test_documents:
            self.test_documents = self._load_test_documents()

        if not self.test_answers:
            self.test_answers = self._load_test_answers()

        assert (len(self.test_documents) == len(self.test_answers)), \
            "You have not enough (or too many) test answers for your documents!"

        logging.info("Loaded test set for dataset %s" % self.name)

        return self.test_documents, self.test_answers

    def load_train(self):
        """
        Loads the training documents and their answers.
        :return: a tuple containing the train documents and the training answers.
        """
        if not self.train_documents:
            self.train_documents = self._load_train_documents()

        if not self.train_answers:
            self.train_answers = self._load_train_answers()

        assert (len(self.train_documents) == len(self.train_answers)), \
            "You have not enough (or too many) train answers for your documents!"

        logging.info("Loaded training set for dataset %s" % self.name)

        return self.train_documents, self.train_answers


class Hulth(Dataset):
    def __init__(self, path):
        super().__init__("Hulth/Answer Set", path)

    def __load_documents(self, folder):
        """
        Loads the documents in the .abstr files contained
        in the specified folder and puts them in a dictionary
        indexed by document id (i.e. the filename without the
        extension).

        :param folder: the folder containing the documents
        :return: a dictionary with the documents
        """

        # This dictionary will contain the documents
        documents = {}

        for doc in os.listdir("%s/%s" % (self.path, folder)):
            if doc.endswith(".abstr"):
                content = open(("%s/%s/%s" % (self.path, folder, doc)), "r").read()
                content = tk.tokenize(content)
                documents[doc[:doc.find('.')]] = content

        return documents

    def __load_answers(self, folder):
        """
        Loads the answers contained in the .contr and .uncontr files
        and puts them in a dictionary indexed by document ID
        (i.e. the document name without the extension)
        :param folder: the folder containing the answer files
        :return: a dictionary with the answers
        """

        # This dictionary will contain the answers
        answers = {}

        for doc in os.listdir("%s/%s" % (self.path, folder)):
            if doc.endswith(".contr") or doc.endswith(".uncontr"):
                content = open(("%s/%s/%s" % (self.path, folder, doc)), "r").read()
                retrieved_answers = content.split(';')
                doc_id = doc[:doc.find('.')]
                for answer in retrieved_answers:
                    answer = answer.strip()
                    tokenized_answer = tk.tokenize(answer)
                    if doc_id not in answers:
                        answers[doc_id] = [tokenized_answer]
                    else:
                        answers[doc_id].append(tokenized_answer)

        return answers

    def _load_test_documents(self):
        return self.__load_documents("Test")

    def _load_train_documents(self):
        return self.__load_documents("Training")

    def _load_test_answers(self):
        return self.__load_answers("Test")

    def _load_train_answers(self):
        return self.__load_answers("Training")

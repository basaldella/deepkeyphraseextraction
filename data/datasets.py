from utils import tokenizer as tk
import logging
import os


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
        self.validation_documents = None
        self.validation_answers = None
        self.tokenizer = tk.tokenizers.keras

        logging.debug("Initialized dataset %s from folder %s" %
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

    def _load_validation_documents(self):
        """
        Loads the validation documents.

        :return: a list of documents.
        """
        raise NotImplementedError

    def _load_validation_answers(self):
        """
        Loads the answers for the validation documents.
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

        logging.debug("Loaded test set for dataset %s" % self.name)

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

        logging.debug("Loaded training set for dataset %s" % self.name)

        return self.train_documents, self.train_answers


    def load_validation(self):
        """
        Loads the validation documents and their answers.
        :return: a tuple containing the validation documents and the training answers.
        """
        if not self.validation_documents:
            self.validation_documents = self._load_validation_documents()

        if not self.validation_answers:
            self.validation_answers = self._load_validation_answers()

        assert (len(self.validation_documents) == len(self.validation_answers)), \
            "You have not enough (or too many) validation answers for your documents!"

        logging.debug("Loaded validation set for dataset %s" % self.name)

        return self.validation_documents, self.validation_answers

class Hulth(Dataset):
    """
    Dataset from Annette Hulth's "Improved Automatic Keyword Extraction
    Given More Linguistic Knowledge"

    Note: to make the results obtained with this dataset comparable to
    the ones described in Hulth's paper, only the "uncontrolled" terms
    are used.

    Full-text here: http://www.aclweb.org/anthology/W03-1028
    """
    def __init__(self, path):
        super().__init__("Hulth, 2003", path)

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
                content = tk.tokenize(content,self.tokenizer)
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
            if doc.endswith(".uncontr"):
                content = open(("%s/%s/%s" % (self.path, folder, doc)), "r").read()
                retrieved_answers = content.split(';')
                doc_id = doc[:doc.find('.')]
                for answer in retrieved_answers:
                    answer = answer.strip()
                    tokenized_answer = tk.tokenize(answer,self.tokenizer)
                    if doc_id not in answers:
                        answers[doc_id] = [tokenized_answer]
                    else:
                        answers[doc_id].append(tokenized_answer)

        return answers

    def _load_test_documents(self):
        return self.__load_documents("Test")

    def _load_train_documents(self):
        return self.__load_documents("Training")

    def _load_validation_documents(self):
        return self.__load_documents("Validation")

    def _load_test_answers(self):
        return self.__load_answers("Test")

    def _load_train_answers(self):
        return self.__load_answers("Training")

    def _load_validation_answers(self):
        return self.__load_answers("Validation")


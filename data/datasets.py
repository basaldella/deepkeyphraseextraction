import logging
import os

from nlp import tokenizer as tk


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

        assert (not self.validation_answers and not self.validation_answers) or \
            (len(self.validation_documents) == len(self.validation_answers)), \
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
                    if doc_id not in answers:
                        answers[doc_id] = [answer]
                    else:
                        answers[doc_id].append(answer)

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


class Semeval2017(Dataset):
    def __init__(self, path):
        super().__init__("Semeval 2017", path)

    def __load_documents(self, folder):
        """
        Loads the documents in the .txt files contained
        in the specified folder and puts them in a dictionary
        indexed by document id (i.e. the filename without the
        extension).

        :param folder: the folder containing the documents
        :return: a dictionary with the documents
        """

        # This dictionary will contain the documents
        documents = {}

        for doc in os.listdir("%s/%s" % (self.path, folder)):
            if doc.endswith(".txt"):
                content = open(("%s/%s/%s" % (self.path, folder, doc)), "r").read()
                documents[doc[:doc.find('.')]] = content

        return documents

    def __load_answers(self, folder):
        '''
        Loads the answers contained in the .ann files
        and puts them in a dictionary indexed by document ID
        (i.e. the document name without the extension).

        Adapted from readAnn() from the official Semeval 2017 Scripts.


        :param folder: the folder containing the answer files
        :return: a dictionary with the answers
        '''

        answers = {}

        file_list = os.listdir("%s/%s" % (self.path, folder))
        for filename in file_list:
            if not filename.endswith(".ann"):
                continue
            file_anno = open(os.path.join("%s/%s" % (self.path, folder), filename), "rU")
            file_text = open(os.path.join("%s/%s" % (self.path, folder), filename.replace(".ann", ".txt")), "rU")
            doc_id = filename[:filename.find('.')]

            answers[doc_id] = []

            # there's only one line, as each .txt file is one text paragraph
            for l in file_text:
                text = l

            for l in file_anno:
                anno_inst = l.strip("\n").split("\t")
                if len(anno_inst) == 3:
                    anno_inst1 = anno_inst[1].split(" ")
                    if len(anno_inst1) == 3:
                        keytype, start, end = anno_inst1
                    else:
                        keytype, start, _, end = anno_inst1
                    if not keytype.endswith("-of"):

                        # look up span in text and print error message if it doesn't match the .ann span text
                        keyphr_text_lookup = text[int(start):int(end)]
                        keyphr_ann = anno_inst[2]
                        if keyphr_text_lookup != keyphr_ann:
                            logging.warning("Spans don't match for anno %s in file %s" % (l.strip(), filename))
                        else:
                            answers[doc_id].append(keyphr_ann)

        return answers

    def _load_test_documents(self):
        return self.__load_documents("test")

    def _load_train_documents(self):
        return self.__load_documents("train")

    def _load_validation_documents(self):
        return self.__load_documents("dev/dev")

    def _load_test_answers(self):
        return self.__load_answers("test")

    def _load_train_answers(self):
        return self.__load_answers("train")

    def _load_validation_answers(self):
        return self.__load_answers("dev/dev")


class Marujo2012(Dataset):
    """
    Dataset from Luís Marujo et al: "Supervised Topical Key Phrase Extraction of News Stories
    using Crowdsourcing, Light Filtering and Co-reference Normalization"

    Full text here: http://www.cs.cmu.edu/~lmarujo/publications/lmarujo_LREC_2012.pdf
    """

    def __init__(self, path):
        super().__init__("Marujo, 2012", path)

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
            if doc.endswith(".txt"):
                content = open(("%s/%s/%s" % (self.path, folder, doc)), "r").read()
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
            if doc.endswith(".key"):
                content = open(("%s/%s/%s" % (self.path, folder, doc)), "r").read()
                retrieved_answers = content.split('\n')
                doc_id = doc[:doc.find('.')]
                for answer in retrieved_answers:
                    answer = answer.strip()
                    if len(answer) > 0:
                        if doc_id not in answers:
                            answers[doc_id] = [answer]
                        else:
                            answers[doc_id].append(answer)

        return answers

    def _load_test_documents(self):
        return self.__load_documents("CorpusAndCrowdsourcingAnnotations/test")

    def _load_train_documents(self):
        return self.__load_documents("CorpusAndCrowdsourcingAnnotations/train")

    def _load_validation_documents(self):
        return self.__load_documents("CorpusAndCrowdsourcingAnnotations/validation")

    def _load_test_answers(self):
        return self.__load_answers("CorpusAndCrowdsourcingAnnotations/test")

    def _load_train_answers(self):
        return self.__load_answers("CorpusAndCrowdsourcingAnnotations/train")

    def _load_validation_answers(self):
        return self.__load_answers("CorpusAndCrowdsourcingAnnotations/validation")


class Semeval2010(Dataset):
    """
    Dataset from the Semeval 2010 keyphrase extraction challenge.
    """

    def __init__(self, path):
        super().__init__("Semeval 2010", path)

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
            if doc.endswith(".txt.final"):
                content = open(("%s/%s/%s" % (self.path, folder, doc)), "r").read()
                documents[doc[:doc.find('.')]] = content

        return documents

    def __load_answers(self, fileName):
        """
        Loads the answers contained in the .contr and .uncontr files
        and puts them in a dictionary indexed by document ID
        (i.e. the document name without the extension)
        :param folder: the folder containing the answer files
        :return: a dictionary with the answers
        """

        # This dictionary will contain the answers
        answers = {}

        content = open("%s/%s" % (self.path,fileName), "r").read()
        document_answers = content.split('\n')

        for doc in document_answers:
            doc_id = doc[:doc.find(' : ')]
            retrieved_answers = (doc[doc.find(' : ')+3:]).split(',')
            for answer in retrieved_answers:
                answer = answer.strip()
                if len(answer) > 0:
                    if doc_id not in answers:
                        answers[doc_id] = [answer]
                    else:
                        answers[doc_id].append(answer)

        return answers

    def _load_test_documents(self):
        return self.__load_documents("test")

    def _load_train_documents(self):
        return self.__load_documents("train")

    def _load_validation_documents(self):
        return self.__load_documents("trial")

    def _load_test_answers(self):
        return self.__load_answers("test_answer/test.combined.stem.final")

    def _load_train_answers(self):
        return self.__load_answers("train_answer/train.combined.final")

    def _load_validation_answers(self):
        return self.__load_answers("trial_answer/trial.combined.final")
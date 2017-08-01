class Dataset(object):
    """
    An abstract class that represents a dataset.
    """

    def __init__(self,name,path):
        self.path = path
        self.name = name
        self.test_documents = None
        self.test_answers = None
        self.train_documents = None
        self.train_answers = None

    def __str__(self):
        return 'Dataset %s located in folder %s' % (self.name,self.path)

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
            self._load_test_documents()

        if not self.test_answers:
            self._load_test_answers()

        return self.test_documents,self.test_answers

    def load_train(self):
        """
        Loads the training documents and their answers.
        :return: a tuple containing the train documents and the training answers.
        """
        if not self.train_documents:
            self._load_train_documents()

        if not self.train_answers:
            self._load_test_answers()

class Hulth(Dataset):

    def __init__(self,path):
        super().__init__("Hulth/sequential",path)

    def _load_test_documents(self):
        return

    def _load_test_answers(self):
        return

    def _load_train_answers(self):
        return

    def _load_train_documents(self):
        return
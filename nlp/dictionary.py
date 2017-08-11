from collections import OrderedDict

class Dictionary(object):
    """Dictionary utility class. This class is a lightweight version of the Keras text preprocessing module
    (see https://github.com/fchollet/keras/blob/master/keras/preprocessing/text.py), designed to work on
    tokens instead of strings.

    This class is used to build a dictionary that can in turn be used to fill an Embedding layer
    with word embeddings.

    Please note that `0` is a reserved index that won't be assigned to any word.
    
    The original keras.preprocessing.text module is licensed under the MIT license.
    """

    def __init__(self, num_words=None):

        self.word_counts = OrderedDict()
        self.word_index = {}
        self.num_words = num_words
        self.document_count = 0

    def fit_on_texts(self, tokenized_documents):

        for document in tokenized_documents:
            self.document_count += 1

            for w in document:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

    def texts_to_sequences(self, texts):
        """
        Transforms each text in texts in a sequence of integers.

        Only top "num_words" most frequent words will be taken into account.

        :param texts: A list of words
        :return: A list of sequences.
        """
        texts_sequences = []
        for text in texts:
            texts_sequences.append(self.token_list_to_sequence(text))
        return texts_sequences

    def token_list_to_sequence(self, tokens):
        """Transforms each text in texts in a sequence of integers.

        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            tokens: A list of texts (strings).

        # Yields
            Yields individual sequences.
        """
        vect = []
        for w in tokens:

                i = self.word_index.get(w)
                if i is not None:
                    if self.num_words and i >= self.num_words:
                        continue
                    else:
                        vect.append(i)
        return vect


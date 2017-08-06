import itertools
import numpy as np


def undo_sequential(documents,output):

    return np.argmax(output,axis=2)


def get_words(docs, selections):
    """
    Gets the words selected in the provided documents.

    :param docs: the document to analyze
    :param selections: the words selected in the documents
    :return: a dictionary with the documents and for each a list of
    the selected words
    """
    i = 0
    obtained_words = {}
    for doc, words in docs.items():
        k = 0
        obtained_words_doc = []
        in_word = False
        for token in selections[i]:
            if token == 1 and k < len(words):
                obtained_words_doc.append([words[k]])
                in_word = True
            elif token == 2 and k < len(words) and in_word:
                obtained_words_doc[len(obtained_words_doc) - 1].append(words[k])
            else:
                in_word = False
            k += 1

        # remove duplicate selections
        obtained_words_doc.sort()
        obtained_words_doc = list(w for w, _ in itertools.groupby(obtained_words_doc))
        obtained_words[doc] = obtained_words_doc
        i += 1

    return obtained_words
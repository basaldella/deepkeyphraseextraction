import itertools
import numpy as np
from nlp import chunker


def undo_sequential(output):
    """
    Transforms a 3D one-hot array of the type (documents,token,category)
    in a 2D array of the type (documents,token_category).

    :param output: a one-hot 3D array
    :return: a 2D array
    """
    return np.argmax(output,axis=2)


def get_words(docs, selections):
    """
    Gets the selected words in the provided documents.

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


def get_valid_patterns(answer_set):
    """
    Remove the answers from a set that do NOT match the keyphrase part-of-speech patterns.

    :param answer_set: a dictionary of documents and tokenized keyphrases
    :return: a dicionary of documents and tokenized keyphrases that match the part-of-speech patterns
    """

    doc_filtered = {}

    for doc, kps in answer_set.items():
        doc_filtered[doc] = []
        for kp in kps:
            for valid_kp in chunker.extract_valid_tokens(kp):
                doc_filtered[doc].append(valid_kp)

    return doc_filtered


import nltk
from nltk.chunk.regexp import *
from nlp import tokenizer as tk


KP_REGEX_1 = "<JJ|NN|NNP|NNS|NNPS>*<NN|NNP|NNS|NNPS|VB|VBG>"
KP_REGEX_2 = "<JJ>?<NN|NNS>+<IN><NN|NNS>"
KP_REGEX_3 = "<JJ|VBN>*<NN|NNS>"


def extract_candidates_from_set(set,tokenizer):
    """
    Generates the candidate keyphrases for a document.

    :param set: the training, test or validation set
    :param tokenizer: which tokenizer to use
    :return: a dictionary where each document is associated with its candidate keyphrases
    """

    candidates = {}
    for doc, str in set.items() :
        candidates[doc] = extract_candidates(str,tokenizer)

    return candidates


def extract_candidates(document,tokenizer):
    """
    Extracts the candidate keyphrases from a string.

    :param document: the string to analyze
    :param tokenizer: the tokenizer to use
    :return: the list of candidate keyphrases for the input document
    """

    tokenized_doc = tk.tokenize(document,tokenizer)
    postagged_doc = nltk.pos_tag(tokenized_doc)

    kp_rule_1 = ChunkRule(KP_REGEX_1,"")
    kp_rule_2 = ChunkRule(KP_REGEX_2, "")
    kp_rule_3 = ChunkRule(KP_REGEX_3, "")

    chunk_parser = RegexpChunkParser([kp_rule_1, kp_rule_2, kp_rule_3],
                                     chunk_label="KP")

    tree = chunk_parser.parse(postagged_doc)

    candidates = []

    for subtree in tree.subtrees():
        if subtree.label() == 'KP':
            candidate = []
            for leaf in subtree.leaves():
                candidate.append(leaf[0])
            candidates.append(candidate)

    return candidates

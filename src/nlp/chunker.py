import nltk
from nltk.chunk.regexp import *
from nlp import tokenizer as tk


KP_REGEX_1 = "<JJ|NN|NNP|NNS|NNPS>*<NN|NNP|NNS|NNPS|VB|VBG>"
KP_REGEX_2 = "<JJ>?<NN|NNS>+<IN><NN|NNS>"
KP_REGEX_3 = "<JJ|VBN>*<NN|NNS>"

noun_phrase_grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*|VBG>}  # Nouns and Adjectives, terminated with Nouns or -ing verbs
        
    KP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
"""

hulth_grammar = r"""
    NBAR:
        {<NN.*|JJ.*>*<NN.*|VBG>}  # Nouns and Adjectives, terminated with Nouns or -ing verbs

    VBPART:
        {<VBG|VBP><NBAR>}       # Verb in participle from, then nouns

    COUNT:
        {<CD><NBAR>}            # Numbers then nouns

    NP:
        {<NBAR><IN><NBAR>}
"""

hulth_labels = ['NP','NBAR','COUNT','VBPART']

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

    return extract_valid_tokens(tk.tokenize(document,tokenizer))


def extract_valid_tokens(tokens):
    """
    Given a list of tokens, returns the subsets of such list which are potential keyphrases according to
    the provided part-of-speech patterns.

    :param document: the token list to analyze
    :return: the list of candidate keyphrases for the input document
    """

    postagged_doc = nltk.pos_tag(tokens)

    kp_rule_1 = ChunkRule(KP_REGEX_1,"")
    kp_rule_2 = ChunkRule(KP_REGEX_2, "")
    kp_rule_3 = ChunkRule(KP_REGEX_3, "")

    #chunk_parser = RegexpChunkParser([kp_rule_1, kp_rule_2, kp_rule_3],
    #                                 chunk_label="KP")

    chunk_parser = RegexpParser(grammar=hulth_grammar)

    tree = chunk_parser.parse(postagged_doc)

    candidates = []

    for subtree in tree.subtrees():
        if subtree.label() in hulth_labels:
            candidate = []
            for leaf in subtree.leaves():
                candidate.append(leaf[0])
            if candidate not in candidates:
                candidates.append(candidate)

    return candidates
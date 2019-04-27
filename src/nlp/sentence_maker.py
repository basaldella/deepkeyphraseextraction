import nltk
# from enum import Enum
# from nltk.corpus import gutenberg
from nlp import tokenizer as tk

# splitter_mode = Enum('default_st', 'regex_st')
SENTENCE_TOKENS_PATTERN = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s'


def sentences_set_old(documents, splitter_mode, tokenizer):

    sentenced_docs = {}
    max_sentences_in_doc = 0
    max_words_in_sentence = 0
    for doc, str in documents.items():
        # default_st = nltk.sent_tokenize
        str = str.replace('\n\t', ' ')
        str = str.replace('\n', '. ')
        str = str.replace('\t', '')
        # sentenced_docs[doc] = default_st(text=str)
        sentenced_docs[doc] = sentence_splitter(str, splitter_mode)
        for sent in sentenced_docs[doc]:
            tokens = tk.tokenize(sent, tokenizer)
            if len(tokens) > max_words_in_sentence:
                max_words_in_sentence = len(tokens)
        if len(sentenced_docs[doc]) > max_sentences_in_doc:
            max_sentences_in_doc = len(sentenced_docs[doc])

    return sentenced_docs, max_sentences_in_doc, max_words_in_sentence


def sentences_set(documents, splitter_mode, tokenizer):

    sentenced_docs = {}
    tokenized_docs = {}
    max_sentences_in_doc = 0
    max_words_in_sentence = 0
    for doc, str in documents.items():
        # default_st = nltk.sent_tokenize
        str = str.replace('\n\t', ' ')
        str = str.replace('\n', '. ')
        str = str.replace('\t', '')
        # sentenced_docs[doc] = default_st(text=str)
        sentenced_docs[doc] = sentence_splitter(str, splitter_mode)
        sentences = []
        for sent in sentenced_docs[doc]:
            tokens = tk.tokenize(sent, tokenizer)
            sentences.append(tokens)
            if len(tokens) > max_words_in_sentence:
                max_words_in_sentence = len(tokens)
        tokenized_docs[doc] = sentences
        if len(sentenced_docs[doc]) > max_sentences_in_doc:
            max_sentences_in_doc = len(sentenced_docs[doc])

    return sentenced_docs, tokenized_docs, max_sentences_in_doc, max_words_in_sentence


def sentence_splitter(string, splitter_mode):

    sentenced_string = ''
    if splitter_mode == 'default_st':
        default_st = nltk.sent_tokenize
        sentenced_string = default_st(text=string)
    elif splitter_mode == 'regex_st':
        regex_st = nltk.tokenize.RegexpTokenizer(pattern=SENTENCE_TOKENS_PATTERN, gaps=True)
        sentenced_string = regex_st.tokenize(string)

    return sentenced_string

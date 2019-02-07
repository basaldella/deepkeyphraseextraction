from enum import Enum
import keras.preprocessing.text
import nltk
nltk.download('averaged_perceptron_tagger')

tokenizers = Enum("Tokenizers","nltk keras")


def tokenize_set(documents,answers,tokenizer):

    tokenized_docs = {}
    for doc, str in documents.items():
        tokenized_docs[doc] = tokenize(str, tokenizer)

    tokenized_answers = {}
    for doc, answers in answers.items():
        for answer in answers :
            if doc not in tokenized_answers:
                tokenized_answers[doc] = [tokenize(answer,tokenizer)]
            else:
                tokenized_answers[doc].append(tokenize(answer,tokenizer))

    return tokenized_docs,tokenized_answers

def tokenize(string,tokenizer = tokenizers.keras):
    """
    Tokenizes a string using the selected tokenizer.
    :param string: the string to tokenize
    :param tokenizer: which tokenizer to use (nltk or keras)
    :return: the list of tokens
    """

    if tokenizer == tokenizers.nltk:
        return nltk.word_tokenize(string.lower())
    elif tokenizer == tokenizers.keras:
        return keras.preprocessing.text.text_to_word_sequence(string)
    else:
        raise NotImplementedError()
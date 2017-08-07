from enum import Enum
import keras.preprocessing.text
import nltk

tokenizers = Enum("Tokenizers","nltk keras")


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
from enum import Enum
import keras.preprocessing.text
import nltk

tokenizers = Enum("Tokenizers","nltk keras")
tokenizer = tokenizers.keras

def tokenize(string):
    """
    Tokenizes a string using the selected tokenizer.
    :param string: the string to tokenize
    :return: the list of tokens
    """

    if tokenizer == tokenizers.nltk:
        return nltk.word_tokenize(string)
    elif tokenizer == tokenizers.keras:
        return keras.preprocessing.text.text_to_word_sequence(string)
    else:
        raise NotImplementedError()
import nltk
from nltk import re

ALLOWED_TAGS_HEAD = "NN*|VBN|VBG|JJ|JJR|JJS|RB|CD"
ALLOWED_TAGS_TAIL = "NN*|VBG|CD|\)"


def clean_string(keyphrase):
    """
    Removes the tokens from the head and the tail of a keyphrase +
    (passed as a token list) that do not match the allowed PoS tags.


    :return: the cleaned keyphrase
    """

    keyphrase_pos = nltk.pos_tag(keyphrase)

    start_re = re.compile(ALLOWED_TAGS_HEAD)
    start = 0

    for start in range(len(keyphrase_pos)):
        if not start_re.match(keyphrase_pos[start][1]):
            start += 1
        else:
            break

    end_re = re.compile(ALLOWED_TAGS_TAIL)
    end = len(keyphrase) - 1

    for end in range(len(keyphrase_pos) - 1,):
        if not end_re.match(keyphrase_pos[end][1]):
            end -= 1
        else:
            break

    return keyphrase[start:end+1]



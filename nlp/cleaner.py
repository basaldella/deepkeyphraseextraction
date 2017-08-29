import nltk

# NLTK uses the Penn Treebank tagset
# See http://www.comp.leeds.ac.uk/amalgam/tagsets/upenn.html
ALLOWED_TAGS_HEAD = ["NN","NNP","NNPS","NNS","VBN","VBG","JJ","JJR","JJS","RB","CD"]
ALLOWED_TAGS_TAIL = ["NN","NNP","NNPS","NNS","VBG","CD",")"]


def clean_tokens(keyphrase):
    """
    Removes the tokens from the head and the tail of a keyphrase +
    (passed as a token list) that do not match the allowed PoS tags.


    :return: the cleaned keyphrase
    """

    keyphrase_pos = nltk.pos_tag(keyphrase)

    start = 0

    for start in range(len(keyphrase_pos)):
        if not keyphrase_pos[start][1] in ALLOWED_TAGS_HEAD:
            start += 1
        else:
            break

    end = len(keyphrase) - 1

    for end in range(len(keyphrase_pos) - 1,start,-1):
        if not keyphrase_pos[end][1] in ALLOWED_TAGS_TAIL:
            end -= 1
        else:
            break

    return keyphrase[start:end+1]



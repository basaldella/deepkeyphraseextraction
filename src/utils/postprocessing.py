import itertools
import numpy as np
from nlp import chunker, cleaner


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


def get_top_words(docs,output,words_limit):
    """
    Gets the selected words in the provided documents.

    :param docs: the document to analyze
    :param output: the output of the network
    :param words_limit: how many words to extract
    :return: a dictionary with the documents and for each a list of
    the selected words
    """

    selections = undo_sequential(output)

    i = 0
    obtained_words = {}
    for doc, words in docs.items():
        k = 0
        obtained_words_doc = []
        obtained_words_weights = []
        in_word = False
        for token in selections[i]:
            if token == 1 and k < len(words):
                obtained_words_doc.append([words[k]])
                obtained_words_weights.append(output[i,k,1])
                in_word = True
            elif token == 2 and k < len(words) and in_word:
                obtained_words_doc[len(obtained_words_doc) - 1].append(words[k])
                obtained_words_weights[len(obtained_words_weights) - 1] = \
                    obtained_words_weights[len(obtained_words_weights) - 1] + \
                    ((output[i,k,2] - obtained_words_weights[len(obtained_words_weights) - 1]) /
                     (len(obtained_words_doc[len(obtained_words_doc) - 1])))

                # We calculate the average at the nth step this way:
                # If A_i is the average at the ith step and x_i is the ith item of the sequence, then
                # A_k = A_{k-1} + ((x_k - A_{k-1}) / k)

            else:
                in_word = False
            k += 1

        if words_limit < len(obtained_words_doc):
            # there are more selections than the limit! cut them

            obtained_words_and_scores = {}
            for index, words in enumerate(obtained_words_doc):
                obtained_words_and_scores[index] = obtained_words_weights[index]

            sorted_words = sorted(obtained_words_and_scores, key=obtained_words_and_scores.__getitem__,reverse=True)


            ok_obtained_words = []
            cur_word = 0
            while len(ok_obtained_words) < words_limit and cur_word < len(sorted_words):
                if obtained_words_doc[sorted_words[cur_word]] not in ok_obtained_words:
                    ok_obtained_words.append(obtained_words_doc[sorted_words[cur_word]])
                cur_word += 1
            obtained_words_doc = ok_obtained_words

        else:
            # just remove duplicate selections
            obtained_words_doc.sort()
            obtained_words_doc = list(w for w, _ in itertools.groupby(obtained_words_doc))

        obtained_words[doc] = obtained_words_doc
        i += 1

    return obtained_words


def get_valid_patterns(answer_set):
    """
    Remove the answers from a set that do NOT match the keyphrase part-of-speech patterns.

    :param answer_set: a dictionary of documents and tokenized keyphrases
    :return: a dictionary of documents and tokenized keyphrases that match the part-of-speech patterns
    """

    doc_filtered = {}

    for doc, kps in answer_set.items():
        filtered_keyphrases = []
        for kp in kps:
            for valid_kp in chunker.extract_valid_tokens(kp):
                filtered_keyphrases.append(valid_kp)
        
        # remove duplicates
        filtered_keyphrases.sort()
        filtered_keyphrases = list(w for w, _ in itertools.groupby(filtered_keyphrases))
        doc_filtered[doc] = filtered_keyphrases

    return doc_filtered


def clean_answers(answer_set):
    """
    Cleans the keyphrases by removing the tokens that are not PoS tagged with the allowed tags.

    :param answer_set: a dictionary of documents and tokenized keyphrases
    :return: a dictionary of documents and their cleaned tokenized keyphrases
    """
    doc_filtered = {}

    for doc, kps in answer_set.items():
        filtered_keyphrases = []
        for kp in kps:
            clean_kp = cleaner.clean_tokens(kp)
            if clean_kp:
                filtered_keyphrases.append(clean_kp)

        # remove duplicates
        filtered_keyphrases.sort()
        filtered_keyphrases = list(w for w, _ in itertools.groupby(filtered_keyphrases))
        doc_filtered[doc] = filtered_keyphrases

    return doc_filtered


def get_answers(candidate_tokens,predict_set,predict_result,dictionary):
    """
    Build the dictionary of the selected answer for a QA-based network.

    :param candidate_tokens: the dictionary of the documents and their candidate KPs
    :param predict_set: the input of the network
    :param predict_result: the output of the network
    :param dictionary: the previously-fit word index
    :return: the dictionary of the selected KPs
    """

    # Here the ideas is: we go through the dictionary of the candidates, we find the corresponding
    # model input, and we add the candidate to the answer set if the model predicted class 1 (i.e. that the candidate
    # was a correct KP

    # First, get the actual predictions:
    if np.shape(predict_result)[1] == 1:
        # If we have just 1 output neuron, reshape and put make the output in 0,1 values
        predictions_flattened = np.round(np.reshape(predict_result,np.shape(predict_result)[0]))
    else:
        # If we're working with categorical output, flatten the (num_samples,2) array to a (num_samples) one
        # This way transform a 2D array e.g. [[0.6,0.4] ... [0.2,0.8]] to a 1D array e.g. [0...1]
        predictions_flattened = np.argmax(predict_result, axis=1)

    i = 0
    answers = {}
    for doc_id, candidate_list in candidate_tokens.items() :
        answers[doc_id] = []
        for candidate in candidate_list:

            # Sanity check: was the order preserved?
            assert candidate == dictionary.tokens_to_words(predict_set[1][i])

            if predictions_flattened[i] == 1 :
                answers[doc_id].append(candidate)

            i += 1

    return answers


def get_top_answers(candidate_tokens,predict_set,predict_result,dictionary,limit):
    """
    Build the dictionary of the selected answer for a QA-based network.

    :param candidate_tokens: the dictionary of the documents and their candidate KPs
    :param predict_set: the input of the network
    :param predict_result: the output of the network
    :param dictionary: the previously-fit word index
    :return: the dictionary of the selected KPs
    """

    # Here the ideas is: we go through the dictionary of the candidates, we find the corresponding
    # model input, and we add the candidate to the answer set if the model predicted class 1 (i.e. that the candidate
    # was a correct KP

    # First, get the actual predictions:
    if np.shape(predict_result)[1] == 1:
        # If we have just 1 output neuron, reshape and put the output in 0,1 values
        predictions_flattened = np.round(np.reshape(predict_result,np.shape(predict_result)[0]))
    else:
        # If we're working with categorical output, flatten the (num_samples,2) array to a (num_samples) one
        # This way transform a 2D array e.g. [[0.6,0.4] ... [0.2,0.8]] to a 1D array e.g. [0...1]
        predictions_flattened = np.argmax(predict_result, axis=1)

    i = 0
    answers = {}
    scores = {}
    for doc_id, candidate_list in candidate_tokens.items() :
        answers[doc_id] = []
        scores[doc_id] = []
        for candidate in candidate_list:

            # Sanity check: was the order preserved?
            assert candidate == dictionary.tokens_to_words(predict_set[1][i])

            if predictions_flattened[i] == 1 :
                answers[doc_id].append(candidate)
                if np.shape(predict_result)[1] == 1:
                    scores[doc_id].append(predict_result[i][0])
                else:
                    scores[doc_id].append(predict_result[i][1])

            i += 1

        if len(answers[doc_id]) > limit :
            answers[doc_id] = [x for _,x in sorted(zip(scores[doc_id],answers[doc_id]),reverse=True)][:limit]

    return answers



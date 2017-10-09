from enum import Enum
from nltk.stem import *

stemMode = Enum("StemmerMode","none both results")


def precision(reference,obtained,stem = stemMode.none):

    true_positives = 0
    false_positives = 0

    for doc, reference_kps_tokens in reference.items():
        obtained_kps_tokens = obtained[doc]

        reference_kps = []
        obtained_kps = []

        for ref_tokens in reference_kps_tokens:

            if stem == stemMode.both:
                stemmer = PorterStemmer()
                ref_tokens = [stemmer.stem(token) for token in ref_tokens]

            reference_kp = ' '.join(ref_tokens)
            reference_kps.append(reference_kp.lower())

        for obt_tokens in obtained_kps_tokens:

            if stem == stemMode.both or stem == stemMode.results:
                stemmer = PorterStemmer()
                obt_tokens = [stemmer.stem(token) for token in obt_tokens]

            obt_string = ' '.join(obt_tokens).lower()
            if obt_string not in obtained_kps:
                # this is necessary, because if we stem the kps we may
                # obtain duplicates
                obtained_kps.append(obt_string)

        for obt_string in obtained_kps:
            if obt_string in reference_kps:
                true_positives += 1
            else:
                false_positives += 1

    return (true_positives * 1.0) / (true_positives + false_positives) if true_positives + false_positives > 0 else 0


def recall(reference,obtained,stem=stemMode.none):

    true_positives = 0
    total_reference = sum(len(kps) for doc,kps in reference.items())

    for doc, reference_kps_tokens in reference.items():
        obtained_kps_tokens = obtained[doc]

        reference_kps = []

        for ref_tokens in reference_kps_tokens:

            if stem == stemMode.both:
                stemmer = PorterStemmer()
                ref_tokens = [stemmer.stem(token) for token in ref_tokens]

            reference_kp = ' '.join(ref_tokens)
            reference_kps.append(reference_kp)

        for obt_tokens in obtained_kps_tokens:

            if stem == stemMode.both or stem == stemMode.results:
                stemmer = PorterStemmer()
                obt_tokens = [stemmer.stem(token) for token in obt_tokens]

            obt_string = ' '.join(obt_tokens)
            if obt_string in reference_kps:
                true_positives += 1
                reference_kps.remove(obt_string)

    return (true_positives * 1.0) / total_reference


def f1(precision, recall):
    return (2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0
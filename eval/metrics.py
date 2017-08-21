import logging

def precision(reference,obtained):

    true_positives = 0
    false_positives = 0

    for doc, reference_kps_tokens in reference.items():
        obtained_kps_tokens = obtained[doc]

        reference_kps = []

        for ref_tokens in reference_kps_tokens:
            reference_kp = ' '.join(ref_tokens)
            reference_kps.append(reference_kp.lower())

        for obt_tokens in obtained_kps_tokens:
            obt_string = ' '.join(obt_tokens)
            if obt_string.lower() in reference_kps:
                true_positives += 1
            else:
                false_positives += 1

    return (true_positives * 1.0) / (true_positives + false_positives) if true_positives + false_positives > 0 else 0


def recall(reference,obtained):

    true_positives = 0
    total_reference = sum(len(kps) for doc,kps in reference.items())

    for doc, reference_kps_tokens in reference.items():
        obtained_kps_tokens = obtained[doc]

        reference_kps = []

        for ref_tokens in reference_kps_tokens:
            reference_kp = ' '.join(ref_tokens)
            reference_kps.append(reference_kp)

        for obt_tokens in obtained_kps_tokens:
            obt_string = ' '.join(obt_tokens)
            if obt_string in reference_kps:
                true_positives += 1

    return (true_positives * 1.0) / total_reference


def f1(precision, recall):
    return (2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0
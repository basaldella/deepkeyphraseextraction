import numpy as np

def keras_precision(y_true,y_pred) :

    true_positives = 0
    false_positives = 0

    # reduce dimensionality
    y_true_2d = np.argmax(y_true,axis=2)
    y_pred_2d = np.argmax(y_pred,axis=2)

    y_true_indices = {}

    for i in range(np.shape(y_true_2d)[0]):
        doc_true_indices = []
        in_word = False

        for j in range(np.shape(y_true_2d)[1]):
            if y_true_2d[i][j] == 1 :
                doc_true_indices.append(["%s" % j])
                in_word = True
            elif j > 0 and y_true_2d[i][j] == 2 and in_word:
                doc_true_indices[len(doc_true_indices) -1].append(",%s" % j)
            else:
                in_word = False

        y_true_indices[i] = doc_true_indices

    y_pred_indices = {}

    for i in range(np.shape(y_pred_2d)[0]):
        doc_true_indices = []
        in_word = False
        for j in range(np.shape(y_pred_2d)[1]):

            if y_pred_2d[i][j] == 1:
                doc_true_indices.append(["%s" % j])
                in_word = True
            elif j > 0 and y_pred_2d[i][j] == 2 and in_word:
                doc_true_indices[len(doc_true_indices) - 1].append(",%s" % j)
            else :
                in_word = False

        y_pred_indices[i] = doc_true_indices

    for i in range(len(y_pred_indices)) :
        for kp in y_pred_indices[i]:
            if kp in y_true_indices[i]:
                true_positives += 1
            else :
                false_positives += 1

    return (1.0 * true_positives) / (true_positives + false_positives)

def keras_recall(y_true,y_pred) :

    true_positives = 0
    false_positives = 0

    # reduce dimensionality
    y_true_2d = np.argmax(y_true,axis=2)
    y_pred_2d = np.argmax(y_pred,axis=2)

    y_true_indices = {}

    for i in range(np.shape(y_true_2d)[0]):
        doc_true_indices = []
        in_word = False

        for j in range(np.shape(y_true_2d)[1]):
            if y_true_2d[i][j] == 1 :
                doc_true_indices.append(["%s" % j])
                in_word = True
            elif j > 0 and y_true_2d[i][j] == 2 and in_word:
                doc_true_indices[len(doc_true_indices) -1].append(",%s" % j)
            else:
                in_word = False

        y_true_indices[i] = doc_true_indices

    y_pred_indices = {}

    for i in range(np.shape(y_pred_2d)[0]):
        doc_true_indices = []
        in_word = False
        for j in range(np.shape(y_pred_2d)[1]):

            if y_pred_2d[i][j] == 1:
                doc_true_indices.append(["%s" % j])
                in_word = True
            elif j > 0 and y_pred_2d[i][j] == 2 and in_word:
                doc_true_indices[len(doc_true_indices) - 1].append(",%s" % j)
            else :
                in_word = False

        y_pred_indices[i] = doc_true_indices

    for i in range(len(y_pred_indices)) :
        for kp in y_pred_indices[i]:
            if kp in y_true_indices[i]:
                true_positives += 1

    return (1.0 * true_positives) / sum(len(kps) for doc,kps in y_true_indices.items())


def keras_f1(y_true,y_pred):
    p = keras_precision(y_true,y_pred)
    r = keras_recall(y_true,y_pred)
    return (2*(p * r)) / (p + r)

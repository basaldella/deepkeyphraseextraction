import keras
import numpy as np
import logging


class MetricsCallback(keras.callbacks.Callback):

    def __init__(self,val_x,val_y):
        self.val_x = val_x
        self.val_y = val_y
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):

        # Predict on the validation data
        y_pred = self.model.predict(self.val_x)

        precision = keras_precision(self.val_y,y_pred)
        recall = keras_recall(self.val_y, y_pred)
        f1 = keras_f1(self.val_y, y_pred)

        print("")
        print("###   Validation Scores   ###")
        print("###")
        print("### Epoch     : %s" % (epoch+1))
        print("### Precision : %.4f" % precision)
        print("### Recall    : %.4f" % recall)
        print("### F1        : %.4f" % f1)
        print("###                       ###")

        self.epoch.append(epoch+1)
        self.history.setdefault("precision", []).append(precision)
        self.history.setdefault("recall", []).append(recall)
        self.history.setdefault("f1", []).append(f1)


class MetricsCallbackQA(keras.callbacks.Callback):

    def __init__(self,val_x,val_y,batch_size = 128):
        self.val_x = val_x
        self.val_y = val_y
        self.epoch = []
        self.history = {}
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):

        # Predict on the validation data
        y_pred = self.model.predict(self.val_x,batch_size=self.batch_size,verbose=1)

        precision = keras_precision_qa(self.val_y,y_pred)
        recall = keras_recall_qa(self.val_y, y_pred)
        f1 = keras_f1_qa(self.val_y, y_pred)

        print("")
        print("###   Validation Scores   ###")
        print("###")
        print("### Epoch     : %s" % (epoch+1))
        print("### Precision : %.4f" % precision)
        print("### Recall    : %.4f" % recall)
        print("### F1        : %.4f" % f1)
        print("###                       ###")

        self.epoch.append(epoch+1)
        self.history.setdefault("precision", []).append(precision)
        self.history.setdefault("recall", []).append(recall)
        self.history.setdefault("f1", []).append(f1)

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

    return (1.0 * true_positives) / (true_positives + false_positives) \
        if true_positives + false_positives > 0 else 0

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
    return (2*(p * r)) / (p + r) if p != 0 and r != 0 else 0


def keras_precision_qa(y_true,y_pred) :

    # Prepare data
    if np.shape(y_pred)[1] == 2:
        # If one-hot prediction...
        y_true = np.argmax(y_true,axis=1)
        y_pred = np.argmax(y_pred,axis=1)

    else:
        # If similarity-based...
        y_pred = np.reshape(y_pred, np.shape(y_true))
        y_pred = np.round(y_pred)

    den = np.count_nonzero(y_pred)

    if den == 0:
        logging.log(logging.WARNING,"Network did not predict any positive sample")
        return 0

    return np.count_nonzero(np.in1d(np.where(y_pred), np.where(y_true))) / den


def keras_recall_qa(y_true,y_pred) :
    # Prepare data
    if np.shape(y_pred)[1] == 2:
        # If one-hot prediction...
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    else:
        # If similarity-based...
        y_pred = np.reshape(y_pred, np.shape(y_true))
        y_pred = np.round(y_pred)

    return np.count_nonzero(np.in1d(np.where(y_true), np.where(y_pred))) / np.count_nonzero(y_true)


def keras_f1_qa(y_true,y_pred):
    p = keras_precision_qa(y_true,y_pred)
    r = keras_recall_qa(y_true,y_pred)
    return (2*(p * r)) / (p + r) if p + r > 0 else 0

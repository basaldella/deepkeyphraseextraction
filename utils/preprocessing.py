import numpy as np
from keras.utils import np_utils


def make_categorical(x):
    """
    Transform a two-dimensional list into a 3-dimensional array. The 2nd
    dimension of the input list becomes a one-hot 2D array, e.g.
    if the input is [[1,2,0],...], the output will be
    [[[0,1,0],[0,0,1],[1,0,0]],...]

    :param x: a 2D-list
    :return: a 3D-numpy array
    """

    # How many categories do we have?
    num_categories = max([item for sublist in x for item in sublist]) + 1

    # Prepare numpy output
    new_x = np.zeros((len(x), len(x[0]), num_categories))

    # Use keras to make actual categorical transformation
    i = 0
    for doc in x:
        new_doc = np_utils.to_categorical(doc,num_classes=num_categories)
        new_x[i] = new_doc
        i += 1

    return new_x

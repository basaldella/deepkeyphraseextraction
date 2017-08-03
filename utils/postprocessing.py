import numpy as np

def undo_sequential(documents,output):

    return np.argmax(output,axis=2)

import matplotlib.pyplot as plt


def plot_accuracy(history) :
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy over epochs')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()


def plot_loss(history) :
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss over epochs')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()


def plot_prf(history) :
    plt.plot(history.history['precision'])
    plt.plot(history.history['recall'])
    plt.plot(history.history['f1'])
    plt.title('P/R/F1 scores on validation set')
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.legend(['Precision', 'Recall', 'F1'], loc='upper left')
    plt.show()
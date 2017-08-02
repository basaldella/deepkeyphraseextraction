from data.datasets import Hulth
from utils import glove
import data.datasets as ds
import logging
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Bidirectional,Dense,Dropout,Embedding,LSTM,TimeDistributed
from keras import regularizers

FILTER = '!"#$%&()*+/:<=>?@[\\]^_`{|}~\t\n'
MAX_DOCUMENT_LENGTH = 550
MAX_VOCABULARY_SIZE = 20000
EMBEDDINGS_SIZE = 50

logging.basicConfig(
    format='%(asctime)s\t%(levelname)s\t%(message)s',
    level=logging.INFO)

data = Hulth("data/Hulth2003")

train_doc, train_answer = data.load_train()
test_doc, test_answer = data.load_test()
train_answer_seq = ds.make_sequential(train_doc,train_answer)
test_answer_seq = ds.make_sequential(test_doc,test_answer)

# Transform the documents to sequence
documents_full = []
train_txts = []
test_txts = []
train_y = []
test_y = []

# re-join the tokens obtained with NLTK
# and split them with Keras' preprocessing tools
# to obtain a word sequence

for key, doc in train_doc.items():
    txt = ' '.join(doc)
    documents_full.append(txt)
    train_txts.append(txt)
    train_y = train_answer_seq[key]
for key,doc in test_doc.items():
    txt = ' '.join(doc)
    documents_full.append(txt)
    test_txts.append(txt)
    test_y = test_answer_seq[key]

tokenizer = Tokenizer(num_words=MAX_VOCABULARY_SIZE, filters=FILTER)
tokenizer.fit_on_texts(documents_full)

logging.info("Dictionary fitting completed. Found %s unique tokens" % len(tokenizer.word_index))

# Now we can prepare the actual input
train_x = tokenizer.texts_to_sequences(train_txts)
test_x = tokenizer.texts_to_sequences(test_txts)

logging.info("Longest training document has %s tokens" % len(max(train_x, key=len)))
logging.info("Longest testing document has %s tokens" % len(max(test_x, key=len)))

train_x = pad_sequences(train_x, maxlen=MAX_DOCUMENT_LENGTH)
test_x = pad_sequences(test_x, maxlen=MAX_DOCUMENT_LENGTH)

# prepare the matrix for the embedding layer
word_index = tokenizer.word_index
embeddings_index = glove.load_glove('', EMBEDDINGS_SIZE)

num_words = min(MAX_VOCABULARY_SIZE, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDINGS_SIZE))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

for word, i in word_index.items():
    if i < 5:
        print(word + "\t:");
        print(embedding_matrix[i]);
    else :
        continue

embedding_layer = Embedding(num_words,
                            EMBEDDINGS_SIZE,
                            weights=[embedding_matrix],
                            input_length=MAX_DOCUMENT_LENGTH,
                            trainable=False)

logging.info("Building model...")
model = Sequential()

model.add(embedding_layer)
#
# model.add(Bidirectional(LSTM(500,activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True))) # il primo parametro sono le "unita": dimensioni dello spazio di output
# model.add(Dropout(0.25))
# model.add(TimeDistributed(Dense(200, activation='relu',kernel_regularizer=regularizers.l2(0.01))))
# model.add(Dropout(0.25))
# model.add(TimeDistributed(Dense(3, activation='softmax')))

logging.info("Compiling the model...")
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'],
              sample_weight_mode="temporal")
print(model.summary())
batch_size = 32

epochs = 1
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)

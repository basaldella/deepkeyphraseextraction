import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
from keras.engine import Layer


def load_elmo(dictionary):
    #words_to_embed = ["dog", "cat", "sloth"]
    words_to_embed = list(dictionary.keys())

    elmo = hub.Module("https://tfhub.dev/google/elmo/2")
    embedding_tensor = elmo(words_to_embed)  # <-- removed other params

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        embedding = sess.run(embedding_tensor)
        print(embedding.shape)

    return embedding

# tutorial: https://medium.com/@joeyism/embedding-with-tensorflow-hub-in-a-simple-way-using-elmo-d1bfe0ada45c


class ElmoEmbeddingLayer(Layer):
    def __init__(self, batch_size, max_doc_length, **kwargs):  # gl: adapting to input
        self.dimensions = 1024
        self.trainable = True
        self.batch_size = batch_size  # gl: adapting to input
        self.max_len = max_doc_length  # gl: adapting to input
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='tokens',
                           )['elmo']
        '''
        result = self.elmo(inputs={
            "tokens": tf.squeeze(tf.cast(x, tf.string)),
            "sequence_len": tf.constant(self.batch_size*[self.max_len])
        },
            as_dict=True,
            signature='tokens',
        )['elmo']
        '''
        result = self.elmo(inputs={
            "tokens": tf.squeeze(tf.cast(x, tf.string)),
            "sequence_len": tf.constant(self.max_len)
        },
            as_dict=True,
            signature='tokens',
        )['elmo']
        '''
        return result

    '''
    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')
    '''

    def compute_output_shape(self, input_shape):
        # return (input_shape[0], self.dimensions)
        return input_shape[0], self.max_len, self.dimensions

# tutorial: https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb


class ELMoEmbedding(Layer):

    def __init__(self, idx2word, output_mode="default", trainable=True, **kwargs):
        assert output_mode in ["default", "word_emb", "lstm_outputs1", "lstm_outputs2", "elmo"]
        assert trainable in [True, False]
        self.idx2word = idx2word
        self.output_mode = output_mode
        self.trainable = trainable
        self.max_length = None
        self.word_mapping = None
        self.lookup_table = None
        self.elmo_model = None
        self.embedding = None
        super(ELMoEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.max_length = input_shape[1]
        self.word_mapping = [x[0] for x in sorted(self.idx2word.items(), key=lambda x: x[1])]
        self.word_mapping.insert(0, '')  # gl: vocabulary starts from value=1
        self.lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(self.word_mapping,
                                                                                default_value="<UNK>")
        self.lookup_table.init.run(session=K.get_session())
        # self.elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.trainable)
        self.elmo_model = hub.Module('https://tfhub.dev/google/elmo/2',
                                     trainable=self.trainable,
                                     name="{}_module".format(self.name))
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ELMoEmbedding, self).build(input_shape)

    def call(self, x, mask=None):
        x = tf.cast(x, dtype=tf.int64)
        sequence_lengths = tf.cast(tf.count_nonzero(x, axis=1), dtype=tf.int32)
        strings = self.lookup_table.lookup(x)
        inputs = {
            "tokens": strings,
            "sequence_len": sequence_lengths
        }
        return self.elmo_model(inputs, signature="tokens", as_dict=True)[self.output_mode]

    def compute_output_shape(self, input_shape):
        if self.output_mode == "default":
            return (input_shape[0], 1024)
        if self.output_mode == "word_emb":
            return (input_shape[0], self.max_length, 512)
        if self.output_mode == "lstm_outputs1":
            return (input_shape[0], self.max_length, 1024)
        if self.output_mode == "lstm_outputs2":
            return (input_shape[0], self.max_length, 1024)
        if self.output_mode == "elmo":
            return (input_shape[0], self.max_length, 1024)

    def get_config(self):
        config = {
            'idx2word': self.idx2word,
            'output_mode': self.output_mode
        }
        return list(config.items())

# tutorial: https://github.com/JHart96/keras_elmo_embedding_layer

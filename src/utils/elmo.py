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


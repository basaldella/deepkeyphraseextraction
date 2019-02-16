import tensorflow_hub as hub
import tensorflow as tf


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
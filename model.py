import tensorflow as tf

def getGeoPredictModel(tfidf_input_dim, doc2vec_input_dim, tfidf_hidden_dim, doc2vec_hidden_dim, output_dim):
    tfidfvecs = tf.keras.Input(shape=(tfidf_input_dim,))
    docvecs = tf.keras.Input(shape=(doc2vec_input_dim,))
    tfidf_hidden = tf.keras.layers.Dense(tfidf_hidden_dim)(tfidfvecs)
    docvec_hidden = tf.keras.layers.Dense(doc2vec_hidden_dim)(docvecs)
    concat = tf.keras.layers.Concatenate(axis=-1)([tfidf_hidden, docvec_hidden])
    relu = tf.keras.layers.ReLU()(concat)
    dense = tf.keras.layers.Dense(output_dim)(relu)
    softmax = tf.keras.layers.Softmax(axis=-1)(dense)
    model = tf.keras.Model(inputs=[tfidfvecs, docvecs], outputs=softmax)
    return model

def getContextModel(doc2vec_input_dim, doc2vec_hidden_dim, output_dim):
    docvecs = tf.keras.Input(shape=(doc2vec_input_dim,))
    docvec_hidden = tf.keras.layers.Dense(units=doc2vec_hidden_dim)(docvecs)
    relu = tf.keras.layers.ReLU()(docvec_hidden)
    dense = tf.keras.layers.Dense(units=output_dim)(relu)
    softmax = tf.keras.layers.Softmax(axis=-1)(dense)
    model = tf.keras.Model(inputs=docvecs, outputs=softmax)
    return model
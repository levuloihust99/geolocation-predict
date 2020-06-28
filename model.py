import tensorflow as tf
from tensorflow.keras import layers
from datetime import datetime, date

# class GeoPredict(tf.keras.Model):
#     def __init__(self,
#                  tfidf_hidden_dim=150,
#                  doc2vec_hidden_dim=30,
#                  output_dim=63,
#                  name='geopredict',
#                  **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.tfidf_prj = layers.Dense(units=tfidf_hidden_dim)
#         self.doc2vec_prj = layers.Dense(units=doc2vec_hidden_dim)
#         self.dense = layers.Dense(units=output_dim)
#
#     def call(self, inputs, training=None, mask=None):
#         tfidfvecs = inputs[0]
#         docvecs = inputs[1]
#         x = self.tfidf_prj(tfidfvecs)
#         y = self.doc2vec_prj(docvecs)
#         concat = layers.Concatenate(axis=-1)([x, y])
#         relu = layers.ReLU()(concat)
#         dense = self.dense(relu)
#         softmax = layers.Softmax(axis=-1)(dense)
#         return softmax

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
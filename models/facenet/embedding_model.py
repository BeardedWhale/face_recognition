from keras import Input, Model
from keras.engine.saving import load_model
from keras.layers import Dense, BatchNormalization, Dropout

from models.facenet import triplet_loss
import tensorflow as tf
import numpy as np

# from models.facenet.facenet import get_triplet_loss
from models.facenet.triplet_loss import batch_hard_triplet_loss


def get_triplet_loss(margin=0.6):
    def triplet_loss(labels, embeddings):
        return batch_hard_triplet_loss(labels, embeddings, margin=margin, squared=False)

    return triplet_loss



class EmbeddingModel:

    def __init__(self, facenet_embedding_size, margin=0.6):
        """
                   _________________________________________________________________
           Layer (type)                 Output Shape              Param #
           =================================================================
           input (InputLayer)        (None, 512)               0
           _________________________________________________________________
           dense (Dense)             (None, 256)               131328
           _________________________________________________________________
           dropout (Dropout)         (None, 256)               0
           _________________________________________________________________
           batch_normalization (Batc (None, 256)               1024
           _________________________________________________________________
           dense (Dense)             (None, 512)               131584
           _________________________________________________________________
           batch_normalization (Batc (None, 512)               2048
           =================================================================
           Total params: 265,984
           Trainable params: 264,448
           Non-trainable params: 1,536
           ________________________________________
        """
        super(EmbeddingModel, self).__init__()
        # global graph
        self.facenet_embedding_size = facenet_embedding_size
        self.graph = tf.get_default_graph()
        self.session = tf.Session()
        with self.graph.as_default():
            with self.session.as_default():
                emb = Input(shape=(self.facenet_embedding_size,), dtype='float32')
                dense_1 = Dense(self.facenet_embedding_size // 2, activation='linear', kernel_regularizer='l2')
                norm = BatchNormalization()
                dense_2 = Dense(self.facenet_embedding_size, activation='linear', kernel_regularizer='l2')
                dropout = Dropout(0.5)
                norm2 = BatchNormalization()

                x = dense_1(emb)
                x = dropout(x)
                x = norm(x)
                x = dense_2(x)
                x = norm2(x)

                model = Model(inputs=emb, outputs=x)
                triplet_loss = get_triplet_loss(margin)
                model.compile(loss=triplet_loss, optimizer='adam', metrics=[])
                self.model = model

    def load(self, file_name: str):
        with self.graph.as_default():
            with self.session.as_default():
                triplet_loss = get_triplet_loss()
                model = load_model(file_name, custom_objects={'triplet_loss': triplet_loss})
                self.model = model
                return model

    def fit(self, X: np.array, Y, epochs=80, validation_split=0.3, batch_size=50):
        with self.graph.as_default():
            with self.session.as_default():
                history = self.model.fit(X, Y, epochs=epochs,
                            batch_size=batch_size, shuffle=False, validation_split=validation_split, verbose=0)
                return history

    def predict(self, x):
        with self.graph.as_default():
            with self.session.as_default():
                return self.model.predict(x)
    def evaluate(self, x, y):
        with self.graph.as_default():
            with self.session.as_default():
                return self.model.evaluate(x, y)

    def save(self, file_name):
        with self.graph.as_default():
            with self.session.as_default():
                self.model.save(file_name)


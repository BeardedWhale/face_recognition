import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from image_utils import prewhiten, crop, flip


def load_facenet_model(model, input_map=None):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')


def load_classifier(classifier_filename):
    with open(classifier_filename, 'rb') as infile:
        (model, class_names) = pickle.load(infile)
        return model, class_names


class FaceRecognizer:

    def __init__(self, config):
        self.facenet_path = config.facenet_path
        self.classifier, self.class_names = load_classifier(config.classifier_path)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            with self.sess:
                load_facenet_model(self.facenet_path)
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

    def get_embedding(self, img):
        with self.graph.as_default():
            with tf.Session(graph=self.graph) as sess:
                emb_array = np.zeros((1, self.embedding_size))
                feed_dict = {self.images_placeholder: img, self.phase_train_placeholder: False}
                emb_array[0, :] = sess.run(self.embeddings, feed_dict=feed_dict)
        return emb_array

    def recognize_face(self, img, box, conf=0.6):
        with self.graph.as_default():
            with tf.Session(graph=self.graph) as sess:
                # Get facenet embeddings
                emb_array = np.zeros((1, self.embedding_size))
                image = self.preprocess_image(img, False, False, 160)
                feed_dict = {self.images_placeholder: image, self.phase_train_placeholder: False}
                emb_array[0, :] = sess.run(self.embeddings, feed_dict=feed_dict)

                # Classify
                predictions = self.classifier.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, self.class_names[best_class_indices[i]], best_class_probabilities[i]))
                label = self.class_names[best_class_indices[i]]
                prob = best_class_probabilities[i]
        if prob < conf:
            label = 'Unknown'
        return label, prob

    def preprocess_image(self, img, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
        image = np.zeros((1, image_size, image_size, 3))
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        image[0, :, :, :] = img
        return image

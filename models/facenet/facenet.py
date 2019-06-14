import math
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from tensorflow.python.platform import gfile
from termcolor import colored
from tqdm import tqdm

from image_utils import preprocess_image, get_image_paths_and_labels, load_data


def load_facenet_model(model, input_map=None):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')


def load_classifier(classifier_filename):
    if os.path.isfile(classifier_filename):
        with open(classifier_filename, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            return model, class_names
    return None, None


class Facenet:

    def __init__(self):
        self.facenet_path = 'models/facenet/model.pb'
        self.image_size = 160
        self.classifier, self.class_names = load_classifier('models/facenet/lfw_classifier_KNN_super.pk')
        if not self.classifier:
            self.train_classifier('', '', 'KNN')
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.Session(graph=self.graph) as sess:
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

    def recognize_face(self, img, conf=0.6):
        with self.graph.as_default():
            with tf.Session(graph=self.graph) as sess:
                # Get facenet embeddings
                emb_array = np.zeros((1, self.embedding_size))
                print(img.shape)
                image = np.array([preprocess_image(img, False, False, 160)])
                print(image.shape)
                feed_dict = {self.images_placeholder: image, self.phase_train_placeholder: False}
                emb_array[0, :] = sess.run(self.embeddings, feed_dict=feed_dict)

                # Novelity check

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

    def train_classifier(self, train_data_path, classifier_path, classifier_type, batch_size=32):
        """
        Trains classifier for embeddings
        :param train_data_path: path to folders with train data
        :param classifier_path: path to store classifier
        :param classifier_type: type of classifier to use [KNN, SVC]
        :return: nothing
        """
        k = 5
        classes, labels, image_paths = get_image_paths_and_labels(train_data_path)
        print(f'Number of classes: {len(classes)}')
        print(f'Number of images: {len(image_paths)}')
        print('Calculating features for images')
        nrof_images = len(image_paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, self.embedding_size))
        print(colored('Training classifier', 'red'))
        with self.graph.as_default():
            with tf.Session(graph=self.graph) as sess:
                for i in tqdm(range(nrof_batches_per_epoch)):
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, nrof_images)
                    paths_batch = image_paths[start_index:end_index]
                    images = load_data(paths_batch, self.image_size)
                    feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(self.embeddings, feed_dict=feed_dict)

        classifier_filename_exp = os.path.expanduser(classifier_path)
        if classifier_type == 'KNN':
            model = KNN(n_neighbors=k)
        else:
            model = SVC(kernel='linear', probability=True)
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(emb_array)
        model.fit(emb_array, labels)
        class_names = list(classes.values())
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)
        with open('/Users/elizavetabatanina/PycharmProjects/FaceNet/facenet/models/svm.pk', 'wb') as outfile:
            pickle.dump((clf), outfile)

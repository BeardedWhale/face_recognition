"""
This module is for work with facenet model: https://github.com/davidsandberg/facenet
It allows to:
    - compute face embeddings
    - classify embeddings
    - train classifier (KNN/SVC)
    - use quantized facenet model for mobile devices

Classification model:
    - NN encodes embeddings to make same persons embeddings separate from each other
     (using tripletloss https://github.com/omoindrot/tensorflow-triplet-loss)
    - KNN/SVC assignes class to face embedding (thresholding unkown faces)

-- use_tflite
    -- use_emb_model
"""
import math
import os
import pickle
import time
from functools import partial

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from tensorflow.python.platform import gfile
from termcolor import colored
from tqdm import tqdm

from models.facenet.triplet_loss import batch_hard_triplet_loss
from utils.image_utils import preprocess_image, get_image_paths_and_labels, load_images, crop_face


def load_facenet_model(facenet_path, input_map=None):
    facenet_exp = os.path.expanduser(facenet_path)
    if (os.path.isfile(facenet_exp)):
        print('Model filename: %s' % facenet_exp)
        with gfile.FastGFile(facenet_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')


def load_classifier(classifier_filename):
    if os.path.isfile(classifier_filename):
        with open(classifier_filename, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            return model, class_names
    return None, None


def load_emb_model(emb_model_path):
    k = 0


class Facenet:
    FACENET_PATH = 'models/facenet/model.pb'
    FACENET_TFLITE_PATH = 'models/facenet/facenet.tflite'
    FACENET_EMB_MODEL_PATH = 'models/facenet/embedding_model.h5'

    def __init__(self, classifier_type, tflite=False):
        # Load Facenet model
        self.tflite = tflite
        self.emb_model = None
        if not self.tflite:
            self.graph = tf.Graph()
            with self.graph.as_default():
                with tf.Session(graph=self.graph) as sess:
                    load_facenet_model(Facenet.FACENET_PATH)
                    self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    self.embedding_size = self.embeddings.get_shape()[1]

        else:
            self.interpreter = tf.contrib.lite.Interpreter(model_path=Facenet.FACENET_TFLITE_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        self.classifier_path = f'models/facenet/face_classifier_{classifier_type}{"_tflite" if tflite else ""}.pk'
        self.classifier_type = classifier_type
        self.image_size = 160
        self.classifier, self.class_names = load_classifier(self.classifier_path)
        if not self.classifier:
            print(colored('You need to pretrain face classifier before applying face recognition', 'red'))
        self.embedding_size = 512

    def get_embedding_tflite(self, img):
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def get_embedding(self, img):
        if self.tflite:
            return self.get_embedding_tflite(img)

        # use simple model
        with self.graph.as_default():
            with tf.Session(graph=self.graph) as sess:
                feed_dict = {self.images_placeholder: img, self.phase_train_placeholder: False}
                emb_array = sess.run(self.embeddings, feed_dict=feed_dict)
        return emb_array

    def recognize_face(self, img, conf=0.7, box=()):
        if not self.classifier:
            print(colored('You need to pretrain face classifier!', 'red'))
            return 'Unkown', 0.0
        if box:
            img = crop_face(img, use_box=True, box=box)
        image = np.array([preprocess_image(img, False, False, 160, do_quantize=self.tflite)])
        if self.tflite:
            emb_array = self.get_embedding_tflite(image)
        else:
            # Get facenet embeddings
            emb_array = self.get_embedding(image)

        # Classify
        predictions = self.classifier.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        label = self.class_names[best_class_indices[0]]
        prob = best_class_probabilities[0]
        if prob < conf:
            label = 'Unknown'
        return label, prob

    def train_classifier(self, train_path, test_path='', **kwargs):
        """
        Trains classifier to classify face features
        :param train_data_path: path to train data
        :param test_data_pth: path to test data
        :param batch_size: size
        :param kwargs:
            - batch_size: size of training batch
            - align faces: align faces while loading dataset
            - use_emb_model: use embedding model to get embeddings
            - n_neighbors: n_neighbors for KNN
        :return:
        """
        batch_size = kwargs.get('batch_size', 64)
        align_faces = kwargs.get('align_faces', False)
        use_emb = kwargs.get('use_emb_model', False)
        n_neighbors = kwargs.get('n_neighbors', 5)
        kernel = kwargs.get('kernel', 'rbf')
        emb_size = self.embedding_size

        if self.emb_model is not None and use_emb:
            emb_size //= 2
        print(emb_size)
        dataset = self._get_dataset_(train_path, emb_size, batch_size=batch_size,
                                     tflite=self.tflite, align_faces=align_faces, use_emb_model=use_emb)

        classifier_filename_exp = os.path.expanduser(self.classifier_path)
        print(f'Training {self.classifier_type} classifier...')
        start = time.time()
        if self.classifier_type == 'KNN':
            model = KNN(n_neighbors=n_neighbors)
        else:  # SVC
            model = SVC(kernel=kernel)
        model.fit(dataset['embeddings'], dataset['labels'])
        print(f'Training time: {time.time() - start}')
        class_names = list(dataset['classes'].values())
        print(f'Saving {self.classifier_type} to {classifier_filename_exp}...')
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Done.')

        # Evaluate model
        if test_path:
            test_dataset = self._get_dataset_(test_path, self.embedding_size, batch_size=batch_size,
                                              tflite=self.tflite, align_faces=align_faces)
            predictions = model.predict_proba(test_dataset['embeddings'])
            best_class_indices = np.argmax(predictions, axis=1)
            accuracy = np.mean(np.equal(best_class_indices, test_dataset['labels']))
            print(f'Evaluation {self.classifier_type} accuracy: {accuracy}')

    def train_embeddings(self, train_path, margin=0.3, test_path='', **kwargs):
        """
        Trains embeddings from facenet embeddings using triplet loss
        :param train_path: train faces embeddings from facenet
        :param margin: margin for triplet loss
        :param test_path: path for test data
            - batch_size: size of training batch
            - epochs: number of epochs to train
            - validation_split: which part of data use for validation
            - align faces: align faces while loading dataset
        :return: nothin
        """
        batch_size = kwargs.get('batch_size', 64)
        epochs = kwargs.get('epochs', 200)
        val_split = kwargs.get('validation_split', 0.2)
        align_faces = kwargs.get('align_faces', False)

        dataset = self._get_dataset_(train_path, self.embedding_size, batch_size=batch_size,
                                     tflite=self.tflite, align_faces=align_faces)
        triplet_loss = partial(batch_hard_triplet_loss, margin=margin)

        # Define embedding model
        dense_1 = Dense(self.embedding_size // 2, activation='relu')
        emb = Input(shape=(self.embedding_size,), dtype='float32')
        x = dense_1(emb)
        model = Model(inputs=emb, outputs=x)
        model.compile(loss=batch_hard_triplet_loss, optimizer='adam', metrics=[])
        model.summary()

        # Train model
        start = time.time()
        model.fit(dataset['embeddings'], dataset['labels'], epochs=epochs, batch_size=batch_size, shuffle=True,
                  validation_split=val_split, verbose=0)
        print(f'Training time: {time.time() - start}')
        if test_path:
            test_dataset = self._get_dataset_(test_path, self.embedding_size, tflite=self.tflite,
                                              align_faces=align_faces)
            loss = model.evaluate(test_dataset['embeddings'], test_dataset['labels'])
            print(f'Loss: {loss}')
        self.emb_model = model
        model.save(Facenet.FACENET_EMB_MODEL_PATH)
        print(f'Saved model to {Facenet.FACENET_EMB_MODEL_PATH}')

    def _get_dataset_(self, data_path, emb_size, batch_size=32, tflite=False, use_emb_model=False, align_faces=False):
        """
        Loads dataset from data_path
        :param data_path:
        :param emb_size: self.embedding_size if not use_emb_model
        :param batch_size: size of batch to load
        :param tflite: if use facenet quantized model
        :param use_emb_model: if use neural net to get embeddings
        :param align_faces: if use aligning of faces
        :return: dataset dict: {classes, labels, image_paths, embeddings}
        """
        dataset = {}
        classes, labels, image_paths = get_image_paths_and_labels(data_path)
        print(f'Number of classes: {len(classes)}')
        print(f'Number of images: {len(image_paths)}')
        print('Calculating features for images...')
        nrof_images = len(image_paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        embeddings = np.zeros((nrof_images, emb_size))
        for i in tqdm(range(nrof_batches_per_epoch)):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = image_paths[start_index:end_index]
            images = load_images(paths_batch, self.image_size, quantize=self.tflite)
            if tflite:
                emb = np.zeros((len(images), emb_size))
                for j in range(len(images)):
                    emb_j = self.get_embedding_tflite(np.array([images[j]]))
                    # use embedding model
                    if use_emb_model and self.emb_model:
                        emb_j = self.emb_model.predict(emb_j)
                    emb[j, :] = emb_j
            else:
                emb = self.get_embedding(images)
            embeddings[start_index:end_index, :] = emb

        dataset['classes'] = classes
        dataset['labels'] = labels
        dataset['image_paths'] = image_paths
        dataset['embeddings'] = embeddings
        return dataset

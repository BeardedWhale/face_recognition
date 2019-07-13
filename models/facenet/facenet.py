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
import itertools
import os
import pickle
import time
import warnings

import numpy as np
import tensorflow as tf
from keras.engine.saving import load_model
from loguru import logger
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from tensorflow.python.platform import gfile

from face_recognition import FaceRecognizer
from models.facenet.embedding_model import EmbeddingModel
from models.facenet.triplet_loss import batch_hard_triplet_loss
from utils.face_base import FaceBase
from utils.image_utils import preprocess_image


def load_facenet_model(facenet_path, input_map=None):
    facenet_exp = os.path.expanduser(facenet_path)
    if os.path.isfile(facenet_exp):
        logger.info('Model filename: %s' % facenet_exp)
        with gfile.FastGFile(facenet_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')


def get_triplet_loss(margin=0.6):
    def triplet_loss(labels, embeddings):
        return batch_hard_triplet_loss(labels, embeddings, margin=margin, squared=False)

    return triplet_loss


def load_classifier(classifier_filename):
    if os.path.isfile(classifier_filename) and os.path.exists(classifier_filename):
        with open(classifier_filename, 'rb') as infile:
            model = pickle.load(infile)
            return model
    return None


def load_emb_model(emb_model_path):
    triplet_loss = get_triplet_loss()
    model = load_model(emb_model_path, custom_objects={'triplet_loss': triplet_loss})
    return model


class Facenet(FaceRecognizer):
    FACENET_PATH = 'models/facenet/model.pb'
    FACENET_TFLITE_PATH = 'models/facenet/facenet.tflite'
    FACENET_EMB_MODEL_PATH = 'models/facenet/embedding_model.h5'
    FACE_BASE_PATH = 'face_base'

    def __init__(self, classifier_type, tflite=False, face_base_path=FACE_BASE_PATH, **kwargs):
        """

        :param classifier_type:
        :param tflite:
        :param kwargs:
            -nrof_train_images: number of image to use for training
            -align_faces: if use face aligner to extract faces
        """
        super().__init__()
        align_faces = kwargs.get('align_faces', True)
        nrof_train_images = kwargs.get('nrof_train_images', 12)
        # Load Facenet model
        self.tflite = tflite
        self.emb_model_path = f'models/facenet/embedding_model{"_tflite" if tflite else ""}.h5'
        self.emb_model = EmbeddingModel(512)
        self.emb_model.load(self.emb_model_path)
        if not self.tflite:
            self.graph = tf.Graph()
            with self.graph.as_default():
                with tf.Session(graph=self.graph) as sess:
                    load_facenet_model(Facenet.FACENET_PATH)
                    self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        else:
            self.interpreter = tf.contrib.lite.Interpreter(model_path=Facenet.FACENET_TFLITE_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        self.classifier_path = f'models/facenet/face_classifier_{classifier_type}{"_tflite" if tflite else ""}.pk'
        self.classifier_type = classifier_type
        self.image_size = 160
        self.classifier = load_classifier(self.classifier_path)
        self.class_names = []
        if not self.classifier:
            warnings.warn('You need to pretrain face classifier before applying face recognition', ResourceWarning)
        self.facenet_embedding_size = 512
        self.facebase = FaceBase(face_base_path, get_embedding=self.get_embedding, quantize=self.tflite,
                                 image_size=160, align_faces=align_faces, nrof_train_images=nrof_train_images)
        self.class_names = self.facebase.classes

    def get_embedding_tflite(self, img):
        """
        :param img: [np.array]
        :return: [embedding]
        """
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def get_embedding(self, img):
        # tflite quantized model
        if self.tflite:
            return self.get_embedding_tflite(img)

        # use simple model
        with self.graph.as_default():
            with tf.Session(graph=self.graph) as sess:
                feed_dict = {self.images_placeholder: img, self.phase_train_placeholder: False}
                emb_array = sess.run(self.embeddings, feed_dict=feed_dict)
        return emb_array

    def recognize_faces(self, faces_img, conf=0.6):
        """
        Classifies given face image according to known faces' names.
        1. Take Facenet embedding
        2. Transforms Facenet embedding using emb_model trained with triplet loss
        for better performance on local base faces
        3. Classifies image using simple classifier
        :param faces_img: np.array. Image should be in rgb, cropped and aligned according to detected face
        :param conf: confidence bound to drop inkown faces
        :return: list of faces labels, list of probabilities
        """
        if not self.classifier:
            logger.warning('You need to pretrain face classifier!')
            return 'Unkown', 0.0
        faces_img = np.array([preprocess_image(img, False, False, 160, do_quantize=self.tflite) for img in faces_img])
        facenet_embedding = np.zeros((len(faces_img), self.facenet_embedding_size))
        # Get facenet embeddings
        if self.tflite:
            for i in range(len(faces_img)):
                facenet_embedding[i, :] = self.get_embedding_tflite([faces_img[i]])
        else:
            facenet_embedding = self.get_embedding(faces_img)

        # Get new embedding
        embedding = self.emb_model.predict(facenet_embedding)
        # Classify
        predictions = self.classifier.predict_proba(embedding)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        names = [self.class_names[index] for index in best_class_indices]
        logger.debug(f'Detected faces: {list(zip(names, best_class_probabilities))}')
        for i in range(len(names)):
            if best_class_probabilities[i] < conf:
                names[i] = 'Unknown'

        return names, best_class_probabilities

    def train_classifier(self, **kwargs):
        """
        Trains classifier to classify face features
        :param kwargs:
            - n_neighbors: n_neighbors for KNN
            - kernel: svc kernel
        :return:
        """
        n_neighbors = kwargs.get('n_neighbors', 5)
        kernel = kwargs.get('kernel', 'rbf')
        logger.info(f'Training {self.classifier_type} classifier...')
        start = time.time()
        if self.classifier_type == 'KNN':
            model = KNN(n_neighbors=n_neighbors)
        else:  # SVC
            model = SVC(kernel=kernel, probability=True)
        train_embeddings = self.emb_model.predict(self.facebase.train_embeddings)
        test_embeddings = self.emb_model.predict(self.facebase.test_embeddings)
        model.fit(train_embeddings, self.facebase.train_labels)
        logger.info(f'Training time: {time.time() - start}')
        classifier_path = f'models/facenet/face_classifier_{self.classifier_type}' \
                          f'{"_tflite" if self.tflite else ""}.pk'
        classifier_filename_exp = os.path.expanduser(classifier_path)
        logger.info(f'Saving {self.classifier_type} to {classifier_filename_exp}...')
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump(model, outfile)

        # Evaluate model
        predictions = model.predict_proba(test_embeddings)
        best_class_indices = np.argmax(predictions, axis=1)
        for i in range(len(best_class_indices)):
            if best_class_indices[i] != self.facebase.test_labels[i]:
                logger.debug(f'false: {self.class_names[best_class_indices[i]]},'
                             f' true: {self.class_names[self.facebase.test_labels[i]]}')
        accuracy = np.mean(np.equal(best_class_indices, self.facebase.test_labels))
        logger.info(f'Evaluation {self.classifier_type} accuracy: {accuracy}')
        self.classifier = model
        return accuracy

    def train_embeddings(self, margin=0.6, batch_size=50, epochs=80, val_split=0.3):
        """
        Trains embeddings from facenet embeddings using triplet loss
        :param margin: margin for triplet loss
        :param batch_size: size of training batch
        :param epochs: number of epochs to train
        :param val_split: which part of data use for validation
        :return: nothing
        """
        model = EmbeddingModel(self.facenet_embedding_size, margin)
        # Training model
        start = time.time()
        history = model.fit(self.facebase.train_embeddings, self.facebase.train_labels, epochs=epochs,
                            validation_split=val_split, batch_size=batch_size)
        logger.info(f'Training emb model time: {time.time() - start}')
        val_loss_history = history.history['val_loss']
        val_loss = [val_loss_history[-1]]
        logger.info(f'Emb model Val Loss: {val_loss[0]}')
        loss = model.evaluate(self.facebase.test_embeddings, self.facebase.test_labels)
        val_loss.append(loss)
        self.emb_model = model
        self.emb_model.save(self.emb_model_path)
        logger.info(f'Saved embedding model to {self.emb_model_path}')
        return val_loss

    def update_model(self, faces, name, epochs=50, val_split=0.3, margin=0.6, need_tuning=True,
                     classifier_type='knn'):
        """
        Updates tripnet embeddings model and KNN for face classification with new user and his faces
        :param name: name of a new person to add to base and classifier
        :param batch_size: size of training batch
        :param epochs: number of epochs to train
        :param val_split: which part of data use for validation
        :param margin: margin for triplet loss
        :param classifier_type: type of a classifier to use for update
        :return: nothing
        """
        if faces:
            self.facebase.update(faces, name)
            self.class_names = self.facebase.classes
        total_start = time.time()
        logger.debug(f'Margin for triplet loss: {margin}')
        # Training embedding net
        losses = self.train_embeddings(margin=margin, val_split=val_split, epochs=epochs)
        logger.info(f'Embeddings triplet losses:{losses}')
        # Training classifier
        train_embeddings = self.emb_model.predict(self.facebase.train_embeddings)
        test_embeddings = self.emb_model.predict(self.facebase.test_embeddings)
        clsf_parameters = {'knn': [4, 5, 6, 7, 8, 9, 10],
                           'svc': list(itertools.product(
                               [1, 2, 4, 6, 8, 10, 12],  # C
                               ['rbf', 'poly', 'linear', 'sigmoid']))  # kernel
                           }

        # Classifier tuning
        tuning_parameters = clsf_parameters.get(classifier_type, [1])
        accs = []
        classifiers = []
        start = time.time()
        classifier = None
        self.classifier_type = classifier_type
        for params in tuning_parameters:
            if classifier_type == 'knn':
                classifier = KNN(n_neighbors=params)
            else:  # classifier_type == 'svc':
                classifier = SVC(C=params[0], kernel=params[1])
            classifier.fit(train_embeddings, self.facebase.train_labels)
            classifiers.append(classifier)
            if not need_tuning:  # get the first classifier
                accs = [1.0]
                break
            predictions = classifier.predict(test_embeddings)
            accs.append(np.mean(np.equal(predictions, self.facebase.test_labels)))
        best_id = np.argmax(accs, axis=0)
        self.classifier = classifiers[best_id]
        logger.info(f'Classifier tuning took: {time.time() - start}')
        logger.info(f'Classifier accuracy:{accs[best_id]}')

        classifier_path = f'models/facenet/face_classifier_{classifier_type}' \
                          f'{"_tflite" if self.tflite else ""}.pk'
        classifier_filename_exp = os.path.expanduser(classifier_path)
        logger.info(f'Saving {classifier_type} to {classifier_filename_exp}...')
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump(classifier, outfile)
        logger.info('Updating models took: {time.time() - total_start}')
        logger.success('Recognition model updated!')

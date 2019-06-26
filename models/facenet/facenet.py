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
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.engine.saving import load_model
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from tensorflow.python.platform import gfile
from termcolor import colored
from tqdm import tqdm

# from face_recognition import FaceRecognizer
from face_recognition import FaceRecognizer
from models.facenet.triplet_loss import batch_hard_triplet_loss
from utils.face_base import FaceBase
from utils.image_utils import preprocess_image, load_images, get_faces_paths_and_labels, get_images_paths


def load_facenet_model(facenet_path, input_map=None):
    facenet_exp = os.path.expanduser(facenet_path)
    if os.path.isfile(facenet_exp):
        print('Model filename: %s' % facenet_exp)
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
        """
        super().__init__()
        nrof_train_images = kwargs.get('nrof_train_images', 12)
        # Load Facenet model
        self.tflite = tflite
        emb_model_path = f'models/facenet/embedding_model{"_tflite" if tflite else ""}.h5'
        self.emb_model = load_emb_model(emb_model_path)
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
        self.classifier= load_classifier(self.classifier_path)
        self.class_names = []
        if not self.classifier:
            print(colored('You need to pretrain face classifier before applying face recognition', 'red'))
        self.facenet_embedding_size = 512
        self.facebase = FaceBase(face_base_path, get_embedding=self.get_embedding, quantize=self.tflite,
                                 image_size=160, align_faces=True, nrof_train_images=nrof_train_images)
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

    def recognize_faces(self, faces_img, conf=0.7):
        """
        Classifies given face image according to known faces' names.
        1. Take Facenet embedding
        2. Transforms Facenet embedding using emb_model trained with triplet loss
        for better performance on local base faces
        3. Classifies image using simple classifier
        :param faces_img: np.array. Image should be in rgb, cropped and aligned according to detected face
        :param conf: confidence bound to drop inkown faces
        :param box: face box in order to align not aligned face
        :return: list of faces labels, list of probabilities
        """
        if not self.classifier:
            print(colored('You need to pretrain face classifier!', 'red'))
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
        print(names)
        for i in range(len(names)):
            if best_class_probabilities[i] < conf:
                names[i] = 'Unknown'
        return names, best_class_probabilities

    def train_classifier(self, train_path, **kwargs):
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
        emb_size = self.facenet_embedding_size

        if self.emb_model is not None and use_emb:
            emb_size //= 2
        print(emb_size)

        print(f'Training {self.classifier_type} classifier...')
        start = time.time()
        if self.classifier_type == 'KNN':
            model = KNN(n_neighbors=n_neighbors)
        else:  # SVC
            model = SVC(kernel=kernel)
        model.fit(self.facebase.train_embeddings, self.facebase.train_labels)
        print(f'Training time: {time.time() - start}')
        classifier_path = f'models/facenet/face_classifier_{self.classifier_type}' \
                          f'{"_tflite" if self.tflite else ""}.pk'
        classifier_filename_exp = os.path.expanduser(classifier_path)
        print(f'Saving {self.classifier_type} to {classifier_filename_exp}...')
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump(model, outfile)
        print('Done.')

        # Evaluate model
        predictions = model.predict_proba(self.facebase.test_embeddings)  # fix for svc
        best_class_indices = np.argmax(predictions, axis=1)
        accuracy = np.mean(np.equal(best_class_indices, self.facebase.test_labels))
        print(f'Evaluation {self.classifier_type} accuracy: {accuracy}')
        return accuracy

    def train_embeddings(self, margin=0.6, **kwargs):
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
        batch_size = kwargs.get('batch_size', 50)
        epochs = kwargs.get('epochs', 80)
        val_split = kwargs.get('validation_split', 0.3)

        triplet_loss = get_triplet_loss(margin)
        # Define embedding model
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
        model.compile(loss=triplet_loss, optimizer='adam', metrics=[])
        # model.summary()
        # Training model
        start = time.time()
        history = model.fit(self.facebase.train_embeddings, self.facebase.train_labels, epochs=epochs,
                            batch_size=50, shuffle=False, validation_split=val_split, verbose=0)
        print(f'Training time: {time.time() - start}')
        val_loss_history = history.history['val_loss']
        val_loss = [val_loss_history[-1]]
        print(f'Val Loss: {val_loss[0]}')
        loss = model.evaluate(self.facebase.test_embeddings, self.facebase.test_labels)
        val_loss.append(loss)
        self.emb_model = model
        model.save(Facenet.FACENET_EMB_MODEL_PATH)
        print(f'Saved model to {Facenet.FACENET_EMB_MODEL_PATH}')
        return val_loss

    def update_model(self, faces, name, **kwargs):
        """
        Updates tripnet embeddings model and KNN for face classification with new user and his faces
        :param new_name: name of a new person to add to base and classifier
        :param kwargs:
            - batch_size: size of training batch
            - epochs: number of epochs to train
            - validation_split: which part of data use for validation
            - align faces: align faces while loading dataset
            - n_neighbors: n_neighbors for KNN
            - margin: margin for triplet loss
            - nrof_train_images: number of images of person to use for training, rest of them for testing
            - classifier_type: type of a classifier to use for update
        :return: True in case of successful update, False otherwise
        """
        if faces:
            self.facebase.update(faces, name)
            self.class_names = self.facebase.classes
        total_start = time.time()
        batch_size = kwargs.get('batch_size', 50)
        epochs = kwargs.get('epochs', 50)
        val_split = kwargs.get('validation_split', 0.2)
        align_faces = kwargs.get('align_faces', False)
        n_neighbors = kwargs.get('n_neighbors', 5)
        kernel = kwargs.get('kernel', 'rbf')
        margin = kwargs.get('margin', 0.6)
        need_tuning = kwargs.get('need_tuning', True)
        classifier_type = kwargs.get('classifier_type', 'knn')
        nrof_train_images = kwargs.get('nrof_train_images', 12)
        start = time.time()
        print('Loading data…')
        print(f'Loading data took:{time.time() - start}')
        # Training embedding net
        losses = self.train_embeddings(dataset=self.facebase, margin=margin,
                                       validation_split=val_split, epochs=epochs)
        print(f'Embeddings triplet losses:{losses}')
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
        print(f'Classifier tuning took: {time.time() - start}')
        print(f'Classifier accuracy:{accs[best_id]}')

        classifier_path = f'models/facenet/face_classifier_{classifier_type}' \
                          f'{"_tflite" if self.tflite else ""}.pk'
        classifier_filename_exp = os.path.expanduser(classifier_path)
        print(f'Saving {classifier_type} to {classifier_filename_exp}...')
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump(classifier, outfile)
        print('Done.')
        print(colored(f'Updating models took: {time.time() - total_start}', 'green'))
        return True
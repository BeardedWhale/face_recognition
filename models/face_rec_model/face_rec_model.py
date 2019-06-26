import os
import pickle

import face_recognition
import numpy as np
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from termcolor import colored

from utils.image_utils import get_image_paths_and_labels


def load_face_encodings(path):
    classes, labels, paths = get_image_paths_and_labels(path)
    encodings_base = {}
    encodings = []
    for i, path in enumerate(paths):
        image = face_recognition.load_image_file(path)
        w, h = image.shape[0], image.shape[1]
        encoding = face_recognition.face_encodings(image, known_face_locations=[[w, h, 0, 0]])
        if not encoding:
            print(colored(f'Cannot load {path} to model', 'red'))
            del labels[i]
            continue
        encodings.append(encoding[0])
    encodings_base['classes'] = classes
    encodings_base['labels'] = labels
    encodings_base['embeddings'] = encodings
    print(f'labels:', labels)
    print(len(labels))
    print('classes:', classes)
    print(len(classes))
    return encodings_base


class FaceRecModel:
    def __init__(self, type):
        self.embeddings_base = load_face_encodings(os.path.join('faces_base', 'base'))
        self.classifier_type = type
        self.classifier_path = f'models/face_rec_model/face_classifier_{self.classifier_type}.pk'

    def recognize_face(self, img, box):
        k = 0

    def train_classifier(self, train_data_path, test_data_path='', k=5, metric='minkowski', kernel='rbf'):
        face_encodings = load_face_encodings(train_data_path)
        face_recognition.compare_faces()
        if self.classifier_type == 'KNN':
            model = KNN(n_neighbors=k, metric=metric)
        elif self.classifier_type == 'GMM':
            model = GMM(len(face_encodings['classes']))
        else:
            model = SVC(kernel=kernel)

        model.fit(face_encodings['embeddings'], face_encodings['labels'])
        self.classifier = model
        classifier_filename_exp = os.path.expanduser(self.classifier_path)
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, face_encodings['classes']), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

        # evaluate
        if test_data_path:
            face_encodings = load_face_encodings(test_data_path)
            if self.classifier_type == 'KNN':
                predictions = self.classifier.predict_proba(face_encodings['embeddings'])
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            else:
                best_class_indices = model.predict(face_encodings['embeddings'])
            accuracy = np.mean(np.equal(best_class_indices, face_encodings['labels']))
            print(colored(f'{self.classifier_type} test accuracy: {accuracy}', 'green'))
            for i in range(len(best_class_indices)):
                true = face_encodings['labels'][i]
                predicted = best_class_indices[i]
                if true != predicted:
                    print(colored(f'{true}:{predicted}', 'red'))
                else:
                    print(f'{true}:{predicted}')


def is_in_base(train_set_path, test_set_path, t, tol):
    train_enc = load_face_encodings(train_set_path)
    test_enc = load_face_encodings(test_set_path)
    encodings = test_enc['embeddings']
    predicted = []
    true_labels = []
    z_label = 6
    for i in range(len(test_enc['embeddings'])):
        match_res = face_recognition.compare_faces(train_enc['embeddings'], encodings[i], tolerance=t)
        label = 0
        true_label = 0 if test_enc['labels'][i] != z_label else 1
        label = compute_probability(match_res, train_enc['labels'], tol)
        predicted.append(label)
        true_labels.append(true_label)
    acc = np.mean(np.equal(predicted, true_labels))
    print('Predicted', predicted)
    print('True', true_labels)
    f1 = f1_score(true_labels, predicted)
    print(acc)
    print(f1)


def compute_probability(mathc_res, base_labels, t):
    class_stats = {}
    for i in range(len(mathc_res)):
        if mathc_res:
            stat = class_stats.get(base_labels[i], 0)
            stat += 1
            class_stats[base_labels[i]] = stat
    probs = {}
    print(class_stats)
    for label, stat in class_stats.items():
        probs[label] = stat / len(base_labels)
        if probs[label] > t:
            return 0
    return 1

import glob
import os
import time
from functools import partial
from typing import Dict

import face_recognition
import numpy as np
from termcolor import colored
from tqdm import tqdm


class FaceRecognizer:
    def __init__(self, config):
        self.name = config.recognizer
        self.recognize_face = self.define_recognition_model(config)
        self.update_model = self.define_update_model()
        self.face_features_base = {}

    def define_recognition_model(self, config):
        """
        Gets face detector object
        :param detector_name:
        :return:
        """
        self.faces_base_path = ''
        self.model = None
        print('Loading face recognition model …')
        start = time.time()
        if self.name == 'facenet':
            from models.facenet.facenet import Facenet
            self.model = Facenet(config.classifier_type, True)
            return partial(self.model.recognize_face, )
        if self.name == 'dlib':
            self.faces_base_path = os.path.join('faces_base', 'dlib')
            self.model = face_recognition

            # loading faces_base
            image_paths = glob.glob(os.path.join(self.faces_base_path, '*.jpg'))
            self.face_features_base = {}
            for path in tqdm(image_paths, total=len(image_paths)):
                face_name = path.replace(self.faces_base_path, '').replace('.jpg', '')[1:]
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)
                if not encoding:
                    print(colored(f'Cannot load {path} to model', 'red'))
                    continue
                self.face_features_base[face_name] = encoding[0]

            print(f'Number of faces in base: {len(self.face_features_base)}')
            return partial(self.recognize_face_dlib, faces_features=self.face_features_base)

    def recognize_face_dlib(self, faces_features: Dict, frame, conf, box):
        label = 'Unknown'

        base_encodings = list(faces_features.values())
        labels = list(faces_features.keys())
        if box:
            (startX, startY, endX, endY) = box
            face_enc = face_recognition.face_encodings(frame, known_face_locations=[(endY, endX, startY, startX)])[0]
        else:
            face_enc = face_recognition.face_encodings(frame)[0]

        matching_results = face_recognition.compare_faces(base_encodings, face_enc, tolerance=0.5)
        face_distances = face_recognition.face_distance(base_encodings, face_enc)
        best_match_index = np.argmin(face_distances)

        distance = 10
        if matching_results[best_match_index]:
            label = labels[best_match_index]
            distance = face_distances[best_match_index]
        return label, distance

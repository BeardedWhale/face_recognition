import glob
import os
import time
from collections import Counter
from functools import partial
from typing import Dict, List

import face_recognition
import numpy as np
from termcolor import colored
from tqdm import tqdm


class FaceRecognizer:
    def __init__(self, config):
        self.name = config.recognizer
        self.recognize_faces = self.define_recognition_model(config)
        # self.update_model = self.define_update_model()
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
            return partial(self.model.recognize_faces, )

    def validate_face(self, faces: List, confidence=0.6):
        """
        Validates personality on several face images for better accuracy
        :param faces: list of face images of a same person
        :param confidence: confidence rate
        :return: name of a person
        """
        print('Validating')
        face_labels = Counter()
        start = time.time()
        faces, confidences = self.recognize_faces(faces, conf=0.6)
        face_labels.update(faces)
        label, count = face_labels.most_common(1)[0]
        conf = count/sum(face_labels.values())
        print(f'Face recognition for {len(faces)} faces took {time.time() - start}')
        if conf > confidence:
            return label, conf
        else:
            return 'Unkown', 0.0


import time
from abc import ABC
from collections import Counter
from typing import List, Tuple

from loguru import logger


class FaceRecognizer(ABC):
    def __init__(self):
        # self.logger = log
        pass

    def recognize_faces(self, faces_img: List, conf=0.7) -> Tuple[List, List]:
        pass

    def update_model(self, faces, name, **kwargs):
        pass

    def validate_face(self, faces: List, confidence=0.6):
        """
        Validates personality on several face images for better accuracy
        :param faces: list of face images of a same person
        :param confidence: confidence rate
        :return: name of a person
        """
        face_labels = Counter()
        start = time.time()
        faces, confidences = self.recognize_faces(faces, conf=0.6)  # TODO add as a paramter
        face_labels.update(faces)

        logger.debug(f'Detected faces: {list(zip(faces, confidences))}')
        label, count = face_labels.most_common(1)[0]
        conf = count / sum(face_labels.values())
        logger.info(f'Face recognition for {len(faces)} faces took {time.time() - start}')
        if conf > confidence:
            return label, conf
        else:
            return 'Unkown', 0.0

    def register_new_face(self, faces, id):
        """
        Adds face to face recognition model
        :param faces: list of cropped, aligned rgb images of faces of single person
        :return: new name in db
        """
        print(f'Seems I do not know person {id}')
        name = input(f'Enter new user name for person {id}: ')
        self.update_model(faces, name)
        return name

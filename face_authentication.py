import time
from collections import Counter
from typing import List, Tuple


class FaceRecognizer:
    def __init__(self):
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
        print('Validating')
        face_labels = Counter()
        start = time.time()
        faces, confidences = self.recognize_faces(faces, conf=0.6)
        face_labels.update(faces)
        label, count = face_labels.most_common(1)[0]
        conf = count / sum(face_labels.values())
        print(f'Face recognition for {len(faces)} faces took {time.time() - start}')
        if conf > confidence:
            return label, conf
        else:
            return 'Unkown', 0.0

    def register_new_face(self, faces):
        """
        Adds face to face recognition model
        :param faces: list of cropped, aligned rgb images of faces of single person
        :return: new name in db
        """
        print('Seems I do not know this person')
        name = input('Enter new user name: ')
        self.update_model(faces, name)
        return name

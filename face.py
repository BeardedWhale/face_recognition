import os

import cv2

from face_authentication import FaceRecognizer
from utils.image_utils import align_faces


class TrackableFace:
    MAX_FAILED = 5  # number of maximum failed recognitions before we stop recognizing this face
    MIN_N_FACES = 4  # minimum number of stored aligned faces to recognize person

    def __init__(self, faceID, centroid, face_size, box):
        """
        Creates Face object
        :param faceID:
        :param centroid:
        :param face_size:
        :param box:
        """
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.faceID = faceID
        self.centroids = [centroid]
        self.curr_box = box
        self.photo_path = f'trackable_faces/{faceID}'
        if not os.path.exists(self.photo_path):
            os.mkdir(self.photo_path)

        self.saved_faces = 0
        self.face_size = face_size  # size of face to store
        self.name = 'Unknown'
        self.prob = 0.0
        self.failed_detections = 0
        self.faces = []

    def authorize(self, frame, centroid, face_recognizer: FaceRecognizer):

        if self.failed_detections >= TrackableFace.MAX_FAILED:
            name = face_recognizer.register_new_face(self)
            self.name, self.prob = name, 1.0
            return None

        if len(self.faces) < TrackableFace.MIN_N_FACES:  # not ready for authentication
            # frame = to_rgb(frame)
            face, success = align_faces(frame, box=self.curr_box)
            if success == 1:
                self.faces.append(face)
            print('Not ready')
            return None

        label, conf = face_recognizer.validate_face(self.faces, confidence=0.6)
        if label == 'Unknown':
            self.faces = []
            self.failed_detections += 1
        else:
            self.name = label
            self.prob = conf
            print(f'Registered face {self.faceID} as {label}')

    def save_face(self, frame):
        """
        Saves face to .jpg file to be used for face authentification
        :param frame: current video frame in rgb
        :param centroid: coordinates of a face center
        :return: nothing
        """
        face, success = align_faces(frame, box=self.curr_box)
        if success == 1:
            self.faces.append(face)
            cv2.imwrite(self.photo_path + f'/{self.saved_faces}.jpg', face)
            self.saved_faces += 1

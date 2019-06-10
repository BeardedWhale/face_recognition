import cv2
import math
import os

class TrackableFace:
    def __init__(self, objectID, centroid, face_size):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        self.photo_path = f'trackable_faces/{objectID}'
        if not os.path.exists(self.photo_path):
            os.mkdir(self.photo_path)

        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.detected = False
        self.num_faces = 0
        self.face_size = face_size


    def save_face(self, frame, centroid):
        frame_shape = frame.shape
        coordinates = []
        for i in range(2):
            assert self.face_size <= frame_shape[i]
            diff = int(math.ceil(self.face_size / 2))
            start = centroid[i] - diff
            end = centroid[i] + diff
            if start < 0:
                start = 0
                end = start + self.face_size
            if end >= frame_shape[i]:
                end = frame_shape[i]
                start = end - self.face_size
            coordinates.extend([start, end])

        startX, endX, startY, endY = coordinates
        cropped_face = frame[startY:endY, startX:endX, :]
        cv2.imwrite(self.photo_path + f'/{self.num_faces}.jpg', cropped_face)
        self.num_faces += 1



from functools import partial

import cv2
import numpy as np


class FaceDetector:
    def __init__(self, config):
        self.name = config.detector
        self.detect = self.define_detection_model(config)

    def define_detection_model(self, config):
        """
        Gets face detector object
        :param detector_name:
        :return:
        """
        if self.name == 'ssd':
            net = cv2.dnn.readNetFromCaffe('models/ssd_model/deploy.prototxt.txt',
                                           'models/ssd_model/res10_300x300_ssd_iter_140000.caffemodel')
            return partial(self.__detect_faces_ssd, model=net)
        if self.name == 'mtcnn':
            from mtcnn.mtcnn import MTCNN
            model = MTCNN()
            return partial(self.__detect_faces_mtcnn, model=model)

    def __detect_faces_ssd(self, model, frame, conf):
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300))  # preprocess image
        model.setInput(blob)
        detections = model.forward()
        faces = []
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > conf:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))
        return faces

    def __detect_faces_mtcnn(self, model, frame, conf):
        frame = cv2.resize(frame, (300, 300))
        # model.setInput(blob)
        face_detections = model.detect_faces(frame)
        faces = []
        for face in face_detections:
            (startX, startY, width, hight) = face['box']
            confidence = face['confidence']
            if confidence > conf:
                faces.append((startX, startY, startX + width, startY + hight))
        return faces

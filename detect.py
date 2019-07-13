import argparse
import os
import sys

import cv2
import imutils
from imutils.video import VideoStream
from loguru import logger

from face import TrackableFace
from face_detection import FaceDetector
from face_tracking import MultipleFaceTracker
from models.facenet.facenet import Facenet


def start_tracking(config):
    """
    Starts object tracking
    :param config:
    :return:
    """
    detection_rate = config.detection_rate
    face_size = config.face_size
    conf = config.detection_conf
    mode = config.mode
    vs = VideoStream(src=0).start()

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    total_frames = 0
    trackable_faces = {}

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a Face
    ct = MultipleFaceTracker(tracker_type='correlation', max_disappeared=40, max_distance=50)
    fd = FaceDetector(config)
    fr = None
    if mode == 2 and config.recognizer == 'facenet':
        fr = Facenet(classifier_type='KNN', tflite=True)
    status = 'Waiting'
    logger.debug('Initialized models')
    while True:
        frame = vs.read()
        if frame is None:
            continue
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]
        tracked_faces = []
        if total_frames % detection_rate == 0:
            status = "Detecting"
            face_boxes = fd.detect(frame=frame, conf=conf)
            ct.update_trackers(rgb, face_boxes)
        else:
            status = "Tracking"
            tracked_faces = ct.get_new_positions(rgb)
        faces = ct.update(tracked_faces)
        faces_info = []
        for (faceID, (centroid, box)) in faces.items():
            face = trackable_faces.get(faceID, None)
            if face is None:
                face = TrackableFace(faceID, centroid, face_size, box)
                trackable_faces[faceID] = face
            else:
                face.curr_box = box

            if face.name == 'Unknown' and mode == 2:
                face.authorize(rgb, fr)

            if total_frames % (detection_rate) == 1 and mode > 0:
                face.save_face(frame)

            text = f"ID {faceID}, {face.name}:{face.prob}"
            faces_info.append({'text': text, 'centroid': centroid})
        for info in faces_info:
            centroid = info['centroid']
            text = info['text']
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        info = [("Status", status)]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        total_frames += 1


if __name__ == '__main__':
    if not os.path.exists('trackable_faces'):
        os.mkdir('trackable_faces')
    if not os.path.exists('face_base'):
        os.mkdir('face_base')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=[0, 1, 2], default=2,
                        help='Modes of detection. 0 - detect&track faces, 1 - same as 2 + save faces,'
                             ' 2 - detect&track&register', type=int)
    parser.add_argument('--tracker', choices=['centroid'], default='centroid')
    parser.add_argument('--detector', choices=['ssd', 'mtcnn'], default='ssd')
    parser.add_argument('--recognizer', choices=['facenet'], default='facenet')

    parser.add_argument('--detection_rate', type=int, default=16)
    parser.add_argument('--face_size', type=int, default=160)
    parser.add_argument('--detection_conf', type=float, default=0.7)
    parser.add_argument('--classifier_type', type=str, default='KNN')
    parser.add_argument('--tflite', type=bool, default=True)
    # parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--log_level', type=int, default=10, choices=[5, 10, 20, 25, 30, 40, 50],
                        help='log levels 5-trace, 10 debug, 20-info, etc.')
    configuration = parser.parse_args()
    logger.remove()
    logger.add(sink=sys.stderr, level=configuration.log_level,
               format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>')

    logger.info(f'MODE: {configuration.mode}')
    logger.info(f'FACE DETECTION MODEL: {configuration.detector}')
    logger.info(f'FACE TRACKING MODEL: {configuration.tracker}')
    logger.info(f'FACE RECOGNITION MODEL: {configuration.recognizer}')
    start_tracking(configuration)

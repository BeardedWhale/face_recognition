import argparse

import cv2
import dlib as dlib
import imutils
from imutils.video import VideoStream
from imutils.video import VideoStream
from imutils.video import FPS
from centroidtracker import CentroidTracker
from functools import partial

from face_detection import FaceDetector
from trackableface import TrackableFace


def start_tracking(config):
    """
    Starts object tracking
    :param config:
    :return:
    """
    detection_rate = config.detection_rate
    face_size = config.face_size
    conf = config.detection_conf
    vs = VideoStream(src=0).start()
    writer = None

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableFace
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = [] # array of trackers
    trackableFaces = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # start the frames per second throughput estimator
    fps = FPS().start()
    # define face detection funciton
    fd = FaceDetector(config)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]
        status = "Waiting"
        tracked_faces = []

        if totalFrames % detection_rate == 0:
            status = "Detecting"
            trackers = []
            face_boxes = fd.detect(frame=frame, conf=conf)
            for (i, (startX, startY, endX, endY)) in enumerate(face_boxes):
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
        else:
            # activate tracking
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                # unpack positions
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                tracked_faces.append((startX, startY, endX, endY))
        objects = ct.update(tracked_faces, frame)
        for (objectID, centroid) in objects.items():
            face = trackableFaces.get(objectID, None)
            if face is None:
                face = TrackableFace(objectID, centroid, face_size)
                trackableFaces[objectID] = face
            if totalFrames % (detection_rate*2) == 1:
                face.save_face(frame, centroid)
            text = f"ID {objectID}"
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
        totalFrames += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tracker', choices=['centroid'], default='centroid')
    parser.add_argument('--detector', choices=['ssd'], default='ssd')
    parser.add_argument('--recognizer', choices=['facenet'], default='facenet')
    parser.add_argument('--detection_rate', type=int, default=24)
    parser.add_argument('--face_size', type=int, default=160)
    parser.add_argument('--detection_conf', type=float, default=0.5)
    parser.add_argument('--prototxt', type=str, default='models/ssd_model/deploy.prototxt.txt')
    parser.add_argument('--detection_model', type=str,
                        default='models/ssd_model/res10_300x300_ssd_iter_140000.caffemodel')
    configuration = parser.parse_args()
    print(configuration)
    start_tracking(configuration)
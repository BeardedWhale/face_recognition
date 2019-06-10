import argparse

import cv2
import imutils
from imutils.video import VideoStream

from face_detection import FaceDetector
from face_tracking import TrackableFace, MultipleFaceTracker


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

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    totalFrames = 0
    trackableFaces = {}

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableFace
    ct = MultipleFaceTracker(tracker_type='correlation', max_disappeared=40, max_distance=50)
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
            face_boxes = fd.detect(frame=frame, conf=conf)
            ct.update_trackers(rgb, face_boxes)
        else:
            status = "Tracking"
            tracked_faces = ct.get_new_positions(rgb)

        faces = ct.update(tracked_faces)
        for (faceID, (centroid, box)) in faces.items():
            # print(centroid, box)
            face = trackableFaces.get(faceID, None)
            if face is None:
                face = TrackableFace(faceID, centroid, face_size, box)
                trackableFaces[faceID] = face
            else:
                face.curr_box = box

            if totalFrames % (detection_rate * 2) == 1:
                face.save_face(frame, centroid)

            text = f"ID {faceID}"
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
    start_tracking(configuration)

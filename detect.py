import argparse

import cv2
import imutils
from imutils.video import FileVideoStream, VideoStream
from termcolor import colored

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
    # vs = VideoStream(src=config.video_src).start()
    vs = FileVideoStream(path='test_videos/keanu.mp4').start()

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    total_frames = 0
    trackable_faces = {}

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableFace
    ct = MultipleFaceTracker(tracker_type='correlation', max_disappeared=40, max_distance=50)
    fd = FaceDetector(config)
    if config.recognizer == 'facenet':
        fr = Facenet(classifier_type='KNN', tflite=True)
    print('Initialized models')

    while True:
        frame = vs.read()
        if frame is None:
            continue
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]
        status = "Waiting"
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

            if face.name == 'Unknown':
                face.authorize(rgb, centroid, fr)

            if total_frames % (detection_rate) == 1:
                face.save_face(frame)

            text = f"ID {faceID}, name {face.name}:{face.prob}"
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--tracker', choices=['centroid'], default='centroid')
    parser.add_argument('--detector', choices=['ssd', 'mtcnn'], default='ssd')
    parser.add_argument('--recognizer', choices=['facenet'], default='facenet')
    parser.add_argument('--detection_rate', type=int, default=6)
    parser.add_argument('--face_size', type=int, default=160)
    parser.add_argument('--detection_conf', type=float, default=0.7)
    parser.add_argument('--prototxt', type=str, default='models/ssd_model/deploy.prototxt.txt')
    parser.add_argument('--detection_model', type=str,
                        default='models/ssd_model/res10_300x300_ssd_iter_140000.caffemodel')
    parser.add_argument('--facenet_path', type=str, default='models/facenet/model.pb')
    parser.add_argument('--classifier_type', type=str, default='KNN')
    parser.add_argument('--video_src', type=int, choices=[0, 1], default=0,
                        help='0 for standard webcam, 1 for usb')
    parser.add_argument('--use_recognition', type=bool, default=False)
    configuration = parser.parse_args()

    print('*' * 30)
    print(colored('FACE DETECTION MODEL:', 'blue'), configuration.detector, '\n')
    print(colored('FACE TRACKING MODEL:', 'blue'), configuration.tracker, '\n')
    print(colored('FACE RECOGNITION MODEL:', 'blue'), configuration.recognizer, '\n')
    print('*' * 30)
    start_tracking(configuration)

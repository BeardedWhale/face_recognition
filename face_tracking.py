from collections import OrderedDict

import dlib
import numpy as np
from scipy.spatial import distance as dist


class MultipleFaceTracker:
    """
    This class if used to track multiple faces
    """

    def __init__(self, tracker_type='correlation', max_disappeared=30, max_distance=50):
        self.next_faceID = 0
        self.faces = OrderedDict()  # stores centroids of object
        self.disappeared = OrderedDict()
        self.trackers = []

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.max_disappeared = max_disappeared

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.max_distance = max_distance
        self.trackers = []
        self.create_tracker = self.define_tracker(tracker_type)

    def register(self, centroid, box):
        """
        Registers new face with new ID and face centroid and box
        """
        self.faces[self.next_faceID] = (centroid, box)
        self.disappeared[self.next_faceID] = 0
        self.next_faceID += 1

    def forget(self, faceID):
        # to forget face ID we delete the object ID from
        # both of our respective dictionaries
        del self.faces[faceID]
        del self.disappeared[faceID]

    def update(self, box_coordinates):
        """
        Updates faces position according to trackers by matching newly detected faces
        and old tracked ones. Faces are matched by distance between old and new centroids
        :param box_coordinates: box coordinates of detected faces
        :return: dict {faceID: (centroid coordinates, box_coordinates)}
        """
        if len(box_coordinates) == 0:

            for faceID in list(self.disappeared.keys()):
                self.disappeared[faceID] += 1

                if self.disappeared[faceID] > self.max_disappeared:
                    self.forget(faceID)
            return self.faces

        input_centroids = np.zeros((len(box_coordinates), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(box_coordinates):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.faces) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], box_coordinates[i])
        else:
            faceIDs = list(self.faces.keys())
            face_centroids = []
            for centroid, box in self.faces.values():
                face_centroids.append(centroid)

            # compute dist between each pair of face centroids and
            # input centroids to match newly detected faces to
            # previously tracked ones
            D = dist.cdist(np.array(face_centroids), input_centroids)

            # find smallest value in each row and then sort row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # find smallest value in each column and sort
            # using row index
            cols = D.argmin(axis=1)[rows]

            # remember which faces we have already matched
            used_rows = set()
            used_cols = set()

            # loop over the combination of the (row, column) index
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    # not same object
                    continue

                # get face id from previously tracked faces
                # match new coordinates with previously tracked face
                faceID = faceIDs[row]
                self.faces[faceID] = (input_centroids[col], box_coordinates[col])
                self.disappeared[faceID] = 0

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:  # if some faces have disappeared

                for row in unused_rows:
                    faceID = faceIDs[row]
                    self.disappeared[faceID] += 1

                    if self.disappeared[faceID] > self.max_disappeared:
                        self.forget(faceID)

            else:  # if some faces are new
                for col in unused_cols:
                    self.register(input_centroids[col], box_coordinates[col])
        return self.faces

    def update_trackers(self, rgb, box_coordinates):
        """
        Creates new trackers for newly detected faces
        :param rgb: current frame
        :param box_coordinates: list of (startX, startY, endX, endY) for each detected face
        :return: nothing
        """
        self.trackers = []
        for (i, (startX, startY, endX, endY)) in enumerate(box_coordinates):
            tracker = self.create_tracker((startX, startY, endX, endY), rgb)
            self.trackers.append(tracker)

    def get_new_positions(self, rgb):
        """
        Get new faces position from current frame
        :param rgb: current frame
        :return: list of box coordinates for faces
        """
        tracked_faces = []
        for tracker in self.trackers:
            tracker.update(rgb)
            pos = tracker.get_position()
            # unpack positions
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            tracked_faces.append((startX, startY, endX, endY))

        return tracked_faces

    def define_tracker(self, tracker_type: str):
        """
        Sets what tracker will be used for object tracking
        :param tracker_type: ['correlation', 'KSF', 'MOSSE', …]
        """
        if tracker_type == 'correlation':
            return self.start_track_correlation

        return lambda x, y: None

    def start_track_correlation(self, coordinates, rgb):
        tracker = dlib.correlation_tracker()
        startX, startY, endX, endY = coordinates
        rect = dlib.rectangle(startX, startY, endX, endY)
        tracker.start_track(rgb, rect)
        return tracker

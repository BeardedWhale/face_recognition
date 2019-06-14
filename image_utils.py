import math
import os
from collections import OrderedDict

import cv2
import numpy as np


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def preprocess_image(img, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    if do_prewhiten:
        img = prewhiten(img)
    img = crop(img, do_random_crop, image_size)
    img = flip(img, do_random_flip)
    return img


def crop_face_box(frame, box, face_size=160):
    startX, startY, endX, endY = box
    startX, startY = max(startX, 0), max(startY, 0)
    side_size = max(endX - startX, endY - startY)
    cropped_face = frame[startY:startY + side_size, startX:startX + side_size, :]
    cropped_face = cv2.resize(cropped_face, dsize=(face_size, face_size),
                              interpolation=cv2.INTER_CUBIC)
    return cropped_face


def crop_face_centroid(frame, centroid, face_size=160):
    frame_shape = frame.shape
    coordinates = []
    for i in range(2):
        assert face_size <= frame_shape[i]
        diff = int(math.ceil(face_size / 2))
        start = centroid[i] - diff
        end = centroid[i] + diff
        if start < 0:
            start = 0
            end = start + face_size
        if end >= frame_shape[i]:
            end = frame_shape[i]
            start = end - face_size
        coordinates.extend([start, end])
    startX, endX, startY, endY = coordinates
    cropped_face = frame[startY:endY, startX:endX, :]
    return cropped_face


def crop_face(frame, face_size=160, use_box=True, centroid=(), box=()):
    """
    Crops image by face centroid or by face box
    :param frame: video frame
    :param face_size: required size of a video
    :param use_box: if need to use face box
    :param centroid: (x,y) of face centroid
    :param box: coordinates of face box: (startX, startY, endX, endY)
    :return: cropped frame (squared)
    """
    if use_box and box:
        return crop_face_box(frame, box, face_size)
    if not use_box and centroid:
        return crop_face_centroid(frame, centroid, face_size)
    return frame


def get_image_paths_and_labels(data_path):
    data_path = os.path.expanduser(data_path)
    inner_folders = os.listdir(data_path)
    labels = []
    classes = OrderedDict()
    paths = []
    inner_folders = [folder for folder in inner_folders if os.path.isdir(os.path.join(data_path, folder))]
    for i, folder in enumerate(inner_folders):
        class_name = folder.replace(data_path, '')
        classes[i] = class_name
        images = os.listdir(os.path.join(data_path, folder))
        paths.extend([os.path.join(data_path, folder, image) for image in images])
        labels.extend([i] * len(images))
    return classes, labels, paths


def load_data(image_paths, image_size):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = cv2.imread(image_paths[i])
        img = to_rgb(img)
        img = preprocess_image(img, do_random_crop=False, do_random_flip=False, image_size=image_size)
        images[i, :, :, :] = img
    return images

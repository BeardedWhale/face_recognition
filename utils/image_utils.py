import json
import math
import os
from collections import OrderedDict

import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('utils/shape_predictor_5_face_landmarks.dat')


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def prewhiten(x):
    dtype = x.dtype
    mean = np.mean(x)
    std = np.std(x)
    size = x.size
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj).astype(np.float32)
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


def preprocess_image(img, do_random_crop, do_random_flip, image_size, do_prewhiten=True, do_quantize=False):
    if do_prewhiten:
        img = prewhiten(img)
    if do_quantize:
        img = quantize(img)
    img = crop(img, do_random_crop, image_size)
    img = flip(img, do_random_flip)
    return img


def quantize(img):
    data = (img.astype(np.float64))
    data = (256) * data
    data[data < 0] = 0
    data[data > 255] = 255
    img = data.astype(np.uint8)
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
        images_paths = [os.path.join(data_path, folder, image) for image in images if '.DS' not in image]
        paths.extend(images_paths)
        labels.extend([i] * len(images_paths))
    return classes, labels, paths

def get_images_paths(folder, nrof_train_images):
    """
    Loads all images from given folder
    :param folder:
    :param nrof_train_images:
    :return:
    """
    images = os.listdir(folder)
    images_paths = [os.path.join(folder, image) for image in images if '.DS' not in image]
    train_images, test_images = images_paths[:nrof_train_images], images_paths[nrof_train_images:]
    return train_images, test_images


def get_faces_paths_and_labels(data_path, nrof_train_images):
    """
    Loads faces dataset, ignores folders that are not in config.json
    :param data_path:
    :param nrof_train_images:
    :return:
    """
    data_path = os.path.expanduser(data_path)
    inner_folders = os.listdir(data_path)
    trian_labels = []
    test_labels = []
    with open(os.path.join(data_path, 'config.json')) as file:
        classes = json.loads(file.read())
        print('classes', classes)
    train_paths = []
    test_paths = []
    inner_folders = [folder for folder in inner_folders if os.path.isdir(os.path.join(data_path, folder))]
    for i, folder in enumerate(inner_folders):
        class_name = folder.replace(data_path, '')
        if class_name not in classes:
            continue
        label = classes.index(class_name)
        curr_folder = os.path.join(data_path, folder)
        train_images, test_images = get_images_paths(curr_folder, nrof_train_images)
        train_paths.extend(train_images)
        trian_labels.extend([label] * len(train_images))
        test_paths.extend(test_images)
        test_labels.extend([label] * len(test_images))
    return classes, trian_labels, train_paths, test_labels, test_paths


def load_images(image_paths, image_size, quantize=False, align=False):
    nrof_samples = len(image_paths)
    images = []
    for i in range(nrof_samples):
        img = cv2.imread(image_paths[i])
        img = to_rgb(img)
        if align:
            img, _ = align_faces(img)
        img = preprocess_image(img, do_random_crop=False, do_random_flip=False,
                               image_size=image_size, do_quantize=quantize)
        images.append(img)
    return np.array(images)


def align_faces(img, box=(0, 0, 0, 0)):
    """
    aligns face in the image
    :param img: rgb image
    :return: 0/1, aligned image
    """

    dets = detector(img, 1)
    faces = dlib.full_object_detections()
    target_face_index = -1
    max_iou = 0
    for i, detection in enumerate(dets):
        startX, startY, endX, endY = detection.left(), detection.top(), detection.right(), detection.bottom()
        curr_iou = iou(box, (startX, startY, endX, endY))
        if curr_iou >= max_iou:
            max_iou = curr_iou
            target_face_index = i
        faces.append(sp(img, detection))

    if len(faces) == 0:
        return img, 0

    if target_face_index >= 0 and max_iou >= 0.5 and box != (0, 0, 0, 0):
        img = dlib.get_face_chip(img, faces[target_face_index], size=160)
        return img, 1
    if faces and box == (0, 0, 0, 0):
        return dlib.get_face_chip(img, faces[0], size=160), 1

    return img, 0


def iou(box1, box2):
    """
    Computes intersection over union for two boxes
    :param box1:
    :param box2:
    :return:
    """

    def box_area(startX, startY, endX, endY):
        return (startY - endY) * (startX - endX)

    startX1, startY1, endX1, endY1 = box1
    startX2, startY2, endX2, endY2 = box2
    startX1, startY1 = max(startX1, 0), max(startY1, 0)
    startX2, startY2 = max(startX2, 0), max(startY2, 0)

    area1 = box_area(startX1, startY1, endX1, endY1)
    area2 = box_area(startX2, startY2, endX2, endY2)
    # intersection
    startX = max(startX1, startX2)
    startY = max(startY1, startY2)
    endX = min(endX1, endX2)
    endY = min(endY1, endY2)
    intersection = box_area(startX, startY, endX, endY)

    return intersection / (area1 + area2 - intersection)

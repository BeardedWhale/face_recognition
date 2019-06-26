import json
import os
import time

import numpy as np
from tqdm import tqdm

from utils.image_utils import get_faces_paths_and_labels, load_images, get_images_paths, preprocess_image


class FaceBase:

    def __init__(self, data_path, get_embedding=None, **kwargs):
        """
        Create object with images&embeddings of known faces that are stored in folder (data_path) in following way:
        -- data_path_folder:
            |
             - face1:
                |
                 - img1.jpg
                 - img2.jpg
                 ...
             - face2:
                |
                ...
             ...
            |
            config.json // stores info about dataset classes
                            (in that order, in which classifier will assign labels)

        :param data_path: path to dataset
        :param get_embedding: function to compute face embeddings if needed
        :param kwargs:
            - align_faces: if images have to be properly cropped
            - image_size: crop size
            - quantize: images in np.uint8 or float64
            - nrof_train_images: number of images to use for training, rest for testing
        """
        self.data_path = data_path
        self.classes = []
        with open(os.path.join(data_path, 'config.json')) as file:
            self.classes = json.loads(file.read())
        if get_embedding:
            self.get_embedding = get_embedding
        else:
            self.get_embedding = None

        # Loading images params
        self.align_faces = kwargs.get('align_faces', False)
        self.image_size = kwargs.get('image_size', 160)
        self.quantize = kwargs.get('quantize', False)
        self.nrof_train_images = kwargs.get('nrof_train_images')
        self.train_labels, self.train_images, self.train_embeddings, self.test_labels, \
        self.test_images, self.test_embeddings = self.load_base()

    def load_base(self):
        """
        Loads dataset
        :return:
        """
        start = time.time()
        print(f'Loading face base from {self.data_path}')
        print(f'Number of classes: {len(self.classes)}')

        train_labels, train_paths, test_labels, test_paths = \
            get_faces_paths_and_labels(self.data_path, self.classes, self.nrof_train_images)

        print(f'Number of train images: {len(train_paths)}')
        print(f'Number of test images: {len(test_paths)}')
        print('Calculating features for images...')
        train_len = len(train_paths)
        images_paths = train_paths + test_paths
        nrof_images = len(images_paths)
        images = load_images(images_paths, image_size=self.image_size, quantize=self.quantize,
                             align=self.align_faces)
        train_images = images[:train_len]
        test_images = images[train_len:]
        embeddings = []
        if self.get_embedding is not None:
            for i in tqdm(range(nrof_images), total=nrof_images):
                embeddings.append(self.get_embedding([images[i]])[0])
        embeddings = np.array(embeddings)
        train_embeddings = embeddings[:train_len]
        test_embeddings = embeddings[train_len:]
        print(f'Loading took: {time.time() - start}')
        return train_labels, train_images, train_embeddings, \
               test_labels, test_images, test_embeddings

    def update(self, faces, name) -> bool:
        """
        Updates base with new name in folder
        Folder self.data_path should already contain subfolder face_name
        with face images
        :param faces: list of cropped and aligned rgb images
        :param name: name of a face, should be unique
        :return: True in case of successful update, False otherwise
        """
        start = time.time()
        class_label = len(self.classes)
        self.classes.append(name)
        faces = [preprocess_image(img, do_random_crop=False, do_random_flip=False,
                               image_size=self.image_size, do_quantize=self.quantize) for img in faces]
        train_images = faces[:self.nrof_train_images]
        test_images = faces[self.nrof_train_images:]
        train_len = len(train_images)
        test_len = len(test_images)
        nrof_images = len(faces)
        embeddings = []
        if self.get_embedding is not None:
            for i in tqdm(range(nrof_images), total=nrof_images):
                embeddings.append(self.get_embedding(faces[i])[0])
        embeddings = np.array(embeddings)
        self.train_embeddings = np.concatenate((self.train_embeddings, embeddings[:train_len]), axis=0)
        self.test_embeddings = np.concatenate((self.test_embeddings, embeddings[train_len:]), axis=0)
        self.train_images.extend(train_images)
        self.test_images.extend(test_images)
        self.train_labels.extend([class_label] * train_len)
        self.test_labels.extend([class_label] * test_len)
        if self.get_embedding is not None:
            assert len(self.train_embeddings) == len(self.train_labels)
            assert len(self.test_embeddings) == len(self.test_labels)
        print(f'Updating face base took {time.time() - start}')
        return True

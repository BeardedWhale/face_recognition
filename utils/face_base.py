import json
import os
import time
from collections import OrderedDict
import cv2
import numpy as np
from tqdm import tqdm

from utils.image_utils import get_faces_paths_and_labels, load_images, get_images_paths, preprocess_image, to_rgb

import pickle
from termcolor import colored
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
            base.pk // stores info about dataset classes
                            (in that order, in which classifier will assign labels)

        :param data_path: path to dataset
        :param get_embedding: function to compute face embeddings if needed
        :param kwargs:
            - align_faces: if images have to be properly cropped
            - image_size: crop size
            - quantize: images in np.uint8 or float64
            - nrof_train_images: number of images to use for training, rest for testing
        """
        self.path = data_path
        self.store_file = 'base.pk'
        self.classes = []
        if get_embedding:
            self.get_embedding = get_embedding
        else:
            self.get_embedding = None

        # Loading images params
        self.align_faces = kwargs.get('align_faces', False)
        self.image_size = kwargs.get('image_size', 160)
        self.quantize = kwargs.get('quantize', False)
        self.nrof_train_images = kwargs.get('nrof_train_images')
        start = time.time()
        self.init_face_base(**kwargs)
        print(colored('[FACE BASE]', 'blue'), f'Loading took: {time.time() - start}')

    def load_base_from_folder(self):
        """
        Loads dataset
        :return:
        """
        print(colored('[FACE BASE]', 'blue'), f'Loading face base from {self.path} folder')
        data_path = os.path.expanduser(self.path)
        inner_folders = os.listdir(self.path)
        inner_folders = [folder for folder in inner_folders if os.path.isdir(os.path.join(data_path, folder))]
        if len(inner_folders) == 0:
            raise Exception(f'Face base folder {data_path} doesn\'t contain any subfolders.\n'
                            f'Check base structure in documentation.')

        classes = OrderedDict()
        for i, folder in enumerate(inner_folders):
            class_name = folder.replace(data_path, '')
            classes[i] = class_name

        self.classes = list(classes.values())
        print(colored('[FACE BASE]', 'blue'))
        print('Face base classes:', self.classes)
        train_labels, train_paths, test_labels, test_paths = \
            get_faces_paths_and_labels(self.path, self.classes, self.nrof_train_images)

        print(f'Number of classes: {len(self.classes)}')
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
        return train_labels, train_images, train_embeddings, \
               test_labels, test_images, test_embeddings

    def update(self, faces, name) -> bool:
        """
        Updates base with new name in folder
        Folder self.path should already contain subfolder face_name
        with face images
        :param faces: list of cropped and aligned rgb images
        :param name: name of a face, should be unique
        :return: True in case of successful update, False otherwise
        """
        start = time.time()
        class_label = len(self.classes)
        self.classes.append(name)
        data_path_exp = os.path.expanduser(self.path)
        new_dir = os.path.join(data_path_exp, name)
        os.mkdir(new_dir)
        for i, face in enumerate(faces):
            face = to_rgb(face)
            cv2.imwrite(os.path.join(new_dir, f'{i}.jpg'), face)

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
                embeddings.append(self.get_embedding([faces[i]])[0])
        embeddings = np.array(embeddings)
        self.train_embeddings = np.concatenate((self.train_embeddings, embeddings[:train_len]), axis=0)
        self.test_embeddings = np.concatenate((self.test_embeddings, embeddings[train_len:]), axis=0)
        self.train_images = np.concatenate((self.train_images, train_images), axis=0)
        if test_images:
            self.test_images = np.concatenate((self.test_images, test_images), axis=0)
        # self.train_images.extend(train_images)
        # self.test_images.extend(test_images)
        self.train_labels.extend([class_label] * train_len)
        self.test_labels.extend([class_label] * test_len)
        if self.get_embedding is not None:
            assert len(self.train_embeddings) == len(self.train_labels)
            assert len(self.test_embeddings) == len(self.test_labels)
        print(colored('[FACE BASE]', 'blue'), f'Updating face base took {time.time() - start}')
        self.save()
        return True


    def init_face_base(self, **kwargs):
        """
        Initializes face base by loading base.pk file or by loading face base from img folders
        :return:
        """
        if not os.path.exists(os.path.join(self.path, self.store_file)):
            self.train_labels, self.train_images, self.train_embeddings, \
            self.test_labels, self.test_images, self.test_embeddings = self.load_base_from_folder()
            self.align_faces = kwargs.get('align_faces', False)
            self.image_size = kwargs.get('image_size', 160)
            self.quantize = kwargs.get('quantize', False)
            self.nrof_train_images = kwargs.get('nrof_train_images')

            self.save()
        else:
            # Load all from pickle file
            with open(os.path.join(self.path, self.store_file), 'rb') as file:
                face_base = pickle.load(file)
            params = face_base['params']
            self.align_faces = params.get('align_faces', False)
            self.image_size = params.get('image_size', 160)
            self.quantize = params.get('quantize', False)
            self.nrof_train_images = params.get('nrof_train_images')

            faces = face_base['faces']

            self.classes = faces['classes']
            self.train_labels, self.train_images, self.train_embeddings, self.test_labels, \
            self.test_images, self.test_embeddings = faces['train_labels'], faces['train_images'],\
                                                     faces['train_embeddings'], faces['test_labels'],\
                                                     faces['test_images'], faces['test_embeddings']

    def save(self):
        """
        Saves face to pickle file
        :return: nothing
        """
        print(colored('[FACE BASE]', 'blue'), 'Saving face base...')
        params = {'align_faces': self.align_faces, 'image_size': self.image_size,
                  'quantize': self.quantize, 'nrof_train_images': self.nrof_train_images}
        faces = {'classes': self.classes, 'train_labels': self.train_labels, 'train_images': self.train_images,
                 'train_embeddings': self.train_embeddings, 'test_labels': self.test_labels, 'test_images': self.test_images,
                 'test_embeddings': self.test_embeddings}
        with open(os.path.join(self.path, self.store_file), 'wb+') as file:
            pickle.dump({'params': params, 'faces': faces}, file)
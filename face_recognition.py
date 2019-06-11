import pickle

import cv2
import os

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
from PIL import Image

from matplotlib import pyplot as plt, cm
import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN

from image_utils import to_rgb, prewhiten, crop, flip

def load_facenet_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
            print('kek')

def load_classifier(classifier_filename):
    # classifier_filename_exp = os.path.expanduser(classifier_filename)
    with open(classifier_filename, 'rb') as infile:
        (model, class_names) = pickle.load(infile)
        return model, class_names


class FaceRecognizer:

    def __init__(self, config):
        self.facenet_path = config.facenet_path
        self.classifier, self.class_names = load_classifier(config.classifier_path)


    def recognize_face(self, img, box, conf=0.6):
        label = 'Unknown'
        plt.imshow(img)
        plt.show()
        prob = 0.0
        with tf.Graph().as_default():
            with tf.Session() as sess:
                load_facenet_model(self.facenet_path)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                emb_array = np.zeros((1, embedding_size))
                image = self.preprocess_image(img, False, False, 160)
                feed_dict = {images_placeholder: image, phase_train_placeholder: False}
                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                predictions = self.classifier.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, self.class_names[best_class_indices[i]], best_class_probabilities[i]))
                label = self.class_names[best_class_indices[i]]
                prob = best_class_probabilities[i]
        if prob < conf:
            label = 'Unknown'
        return label, prob




    def preprocess_image(self, img, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
        image = np.zeros((1, image_size, image_size, 3))
        # if img.ndim == 2:
        #     img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        plt.imshow(img)
        plt.show()
        image[0, :, :, :] = img
        return image


# if __name__ == '__main__':
#     fr = FaceRecognizer('/Users/elizavetabatanina/Google Drive/Projects/Face authentification/face_rec_repo/face_recognition/models/facenet/model.pb',
#                         '/Users/elizavetabatanina/Google Drive/Projects/Face authentification/face_rec_repo/face_recognition/models/facenet/lfw_classifier_KNN_7.pk')
#
#     face_path = 'dima.jpg'
#
#     image = Image.open(face_path)
#     # face = cv2.imread(face_path)
#     image = np.asarray(image)
#     fr.recognize_face(image, 0)
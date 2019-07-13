import argparse
import sys

from loguru import logger

from models.facenet.facenet import Facenet


def train_facenet(config):
    facenet = Facenet(config.classifier_type, config.tflite, config.data_path,
                      nrof_train_images=config.nrof_train_images)
    facenet.train_embeddings(config.margin)
    facenet.train_classifier(n_neighbors=config.n_neighbors, kernel=config.kernel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='face_base')
    parser.add_argument('--tflite', type=bool, default=True)
    parser.add_argument('--classifier_type', type=str, choices=['KNN', 'SVC'], default='KNN')
    parser.add_argument('--nrof_train_images', type=int, default=8,
                        help='Number of images to use for training, rest for testing')
    parser.add_argument('--n_neighbors', type=int, default=5)
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--margin', type=float, default=0.6, help='margin for triplet loss')
    parser.add_argument('--log_level', type=int, default=10, choices=[5, 10, 20, 25, 30, 40, 50],
                        help='log levels 5-trace, 10 debug, 20-info, etc.')
    configuration = parser.parse_args()
    logger.remove()
    logger.add(sink=sys.stderr, level=configuration.log_level)

    train_facenet(configuration)

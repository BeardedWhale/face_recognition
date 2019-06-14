from functools import partial


class FaceRecognizer:
    def __init__(self, config):
        self.name = config.recognizer
        self.recognize_face = self.define_recognition_model()

    def define_recognition_model(self):
        """
        Gets face detector object
        :param detector_name:
        :return:
        """
        if self.name == 'facenet':
            from models.facenet.facenet import Facenet
            model = Facenet()
            return partial(model.recognize_face, )

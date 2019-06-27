# Facial recognition system
This code is for face recognition system that allows to add new faces to face base in real time.

## Description

This system consists of 3 major modules:

- Face Detector
- Face Tracker
- Face Recognizer

For each module could be extended by adding new better algorithm. 
### Face Detector
For face detection two models are provided: SSD (with ResNet as backbone) and [MTCNN](https://github.com/ipazc/mtcnn). 

### Face Tracking
For face tracking [centroid tracker](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/) was used.
Future updates: add kalman tracker

### Face Recognition
Face recognition module utilizes [Facenet](https://github.com/davidsandberg/facenet)-based model with embedding size 512.

Facenet model was quantize to make it run on smaller devices like phone. On top of facent simple NN (embedding model) was trained on known
images using [triplet loss](https://github.com/omoindrot/tensorflow-triplet-loss) for better embeddings.
For face classification KNN or SVM are used.

#### Face registration
Module allows to register new face if model understands it doesn't know it. It will ask to enter a new name and 
will retrain model to classify new person

## How to
1. You need to train embedding model and classifier before starting face recognition:
    run ```python train_facenet.py --data_path {your_path to folder} --tflite {it to use tflite facenet} --classifier_type{'KNN'/'SVC'} ```
     see all parameters in train_facenet.py
2. Run ``` python detect.py ``` to start tracking and recognizing faces from webcam. 

    **Arguments**
    
    - --mode     [0, 1, 2]               // mode of detection  0 - detect&track faces, 1 - same as 2 + save faces,'
                                 ' 2 - detect&track&register'
    - --tracker  ['centroid']   // tracking algorithm
    - --detector ['ssd', 'mtcnn']        // face detection model (TODO MTCNN)
    - --recognizer ['facenet'] // which model to use to recognize faces from base
    - --detection_rate          // frame period between running face detection for more accurate face tracking
    - --face_size.             // size of face crop to store in .jpg file
    - --detecttion_conf        // minimum probability (from face detector) to consider object a face
    - --classifier_type [KNN, SVC] // type of classifier to use for face recognition
    
![Demo](https://github.com/BeardedWhale/face_recognition/blob/master/demo/demo1.mov)
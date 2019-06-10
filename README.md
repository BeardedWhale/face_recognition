# face_recognition
Face recognition system

## Run
''' python detect.py '''
### Arguments:
- --tracker  ['centroid']   // tracking algorithm
- --detector ['ssd']        // face detection model (TODO MTCNN)
- --recognizer ['facenet'] // which model to use to recognize faces from base
- --detection_rate          // frame period between running face detection for more accurate face tracking
- --face_size.             // size of face crop to store in .jpg file
- --detecttion_conf        // minimum probability (from face detector) to consider object a face
- --prototxt               // file for ssd detection model
- --detection_model        // file with ssd caffemodel

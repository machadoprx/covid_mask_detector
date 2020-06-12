import cv2

def face_detect_viola_jones(im, factor, min_size, min_neighbors):
    face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
            im, scaleFactor=factor, minSize=min_size,minNeighbors=min_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE)
    return faces
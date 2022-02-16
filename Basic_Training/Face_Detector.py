import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

capture = cv2.VideoCapture(0)

while True:
    ret, image_colr = capture.read()
    image_gray = cv2.cvtColor(image_colr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_colr, scaleFactor=1.2, minNeighbors=10)
    for x, y, w, h in faces:
        cv2.rectangle(image_colr, (x, y), (x + w, y + h), (255, 0, 0), 3)
        roi_gray = image_gray[y:y+h, x:x+w]
        roi_colr = image_colr[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_colr, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 4)

    cv2.imshow('Image', image_colr)
    key = cv2.waitKey(10)

    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

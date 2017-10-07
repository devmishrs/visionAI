'''
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('/opt/opencv/data/haarcascade/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret ,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)
    cv2.imshow('face',img)
    cv2.waitKey(20)
    cv2.destroyAllWindows()
'''

from __future__ import print_function #compactble with python 2 and 3
import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier(filename='/home/dev/github/I_inAI/haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(filename='/home/dev/github/I_inAI/haarcascade/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0,),2)
        roi_gray = gray[y:y+h , x:x+w]
        roi_img = img[y:y+h , x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_img,(ex,ey),(ex+ew , ey+eh), (0,255,0),2)

        cv2.imshow('img',img)
        cv2.waitKey(30)
        cv2.destroyAllWindows()

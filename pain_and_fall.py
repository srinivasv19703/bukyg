import cv2
import os
import time
import numpy as np
from keras.models import load_model
from time import sleep
# from tensorflow.keras.utils import img_to_array
from keras.utils import img_to_array
#from tensorflow.keras.utils import img_to_array
#from keras_preprocessing.image import img_to_array
from keras.preprocessing import image
from twilio.rest import Client
#from keras.preprocessing.image import img_to_array

SID = 'AC3aeab90f7e74f12889797b09915c7130'
token = '89883da40731d36e27c6c23271381c73'

face_classifier = cv2.CascadeClassifier(r"C:\Users\91944\Downloads\AI_Pain_fall_Monitor\AI_Pain_fall_Monitor\haarcascade_frontalface_default.xml")
classifier = load_model(r"C:\Users\91944\Downloads\AI_Pain_fall_Monitor\AI_Pain_fall_Monitor\model.h5")

emotion_labels = ['pain', 'Disgust', 'Fear',
    'Happy', 'Neutral', 'sad', 'Surprise']

fitToEllipse = False

cap = cv2.VideoCapture(0)

im = cap.read()

fgbg = cv2.createBackgroundSubtractorMOG2()
j = 0

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    fgmask = fgbg.apply(gray)

    contours, _ = cv2.findContours(
        fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # List to hold all areas
        areas = []
        for contour in contours:
            ar = cv2.contourArea(contour)
            areas.append(ar)
        max_area = max(areas, default=0)
        max_area_index = areas.index(max_area)
        cnt = contours[max_area_index]
        M = cv2.moments(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(fgmask, [cnt], 0, (255, 255, 255), 3, maxLevel=0)
        if h < w:
            j += 1
        if j > 10:
            print("FALL")
                # cv2.putText(fgmask, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'FALL', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2);print("Need Help")
            # ct=Client(SID,token)
            # ct.messages.create(body="Nurse the Patient was Fallen need some Help! urgently",from_='+15676777643',to='+919943214959')
        if h > w:
            j = 0
            cv2.rectangle(frame, (x, y), (x+w, y+h),(0,0,0),2)  #black
            cv2.putText(frame, 'STRAIGHT', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


#Pain Detection

    for (x, y, w, h) in faces:  # each face
        cv2.rectangle(frame, (x, y), (x+w, y+h),(0,255,255),2) # yellow rectange width and color
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y-15)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
            if label == "pain":
                print("Need Help")
                # ct=Client(SID,token)
                # ct.messages.create(body="Hello Doctor the patient Raghav is in pain need some help",from_='+15676777643',to='+919943214959')
                break
        else:
            cv2.putText(frame, 'No Faces found', (30, 80), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow('AI Patient Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

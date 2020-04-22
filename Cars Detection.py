import cv2
import numpy as np
import time

#Create our body classifier
car_classifier=cv2.CascadeClassifier('./HL_Filters/cars.xml')

#Initiate video capturefor video file
cap=cv2.VideoCapture("./Sample Videos/cars1.mp4")

#Loop once video is Fully Loaded
while cap.isOpened():
    time.sleep(0.1)
    # Read first frame
    ret, frame=cap.read()
    frame=cv2.resize(frame, None,fx=0.5,fy=0.5, interpolation=cv2.INTER_LINEAR)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    cars = car_classifier.detectMultiScale(gray,1.2,3)

    #Extract bounding boxes for any bodies identified
    for(x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0, 255,255),2)
        cv2.imshow("Cars",frame)

    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()
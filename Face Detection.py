import cv2

face_classifier=cv2.CascadeClassifier("./HL_Filters/haarcascade_frontalface_default.xml")

image=cv2.imread("./Sample Images/trump.jpeg")

faces=face_classifier.detectMultiScale(image,1.3,5)

if faces is ():
    print(" No faces detected")

for(x,y,w,h) in faces:

    cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),3)
    cv2.imshow(" Face Detection",image)
    if(cv2.waitKey(0)==13):
         break

cv2.destroyAllWindows()

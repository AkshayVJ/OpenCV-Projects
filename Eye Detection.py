import cv2


face_classifier = cv2.CascadeClassifier("./HL_Filters/haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("./HL_Filters/haarcascade_eye.xml")
image = cv2.imread("./Sample Images/lena.jpg")

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(image, 1.3, 5)

if faces is ():
    print(" No Eyes detected")

for (x, y, w, h) in faces:
    #cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 3)
    #cv2.imshow(" Face Detection",image)
    roi_gray=gray_image[y:y+h,x:x+w]
    roi_color=image[y:y+h,x:x+w]
    eyes= eye_classifier.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 3)
     cv2.imshow(" Eye Detection",image)
    if (cv2.waitKey(0) == 13):
         break

cv2.destroyAllWindows()
import cv2

img1=cv2.imread("./Sample Images/image.jpg")  # input the template file here

cap=cv2.VideoCapture(0)  # 0 for primary camera 1 for secondary camera

while(True):

    ret, frame = cap.read()  # capture frame by frame
    height, width = frame.shape[:2] # Collects height and width from Image
    top_left_x = int(width / 3)
    top_left_y = int((height / 2) + (height / 4))
    bottom_right_x = int((width / 3) * 2)
    bottom_right_y = int((height / 2) - (height / 4))
    # To draw Rectangle window on the screen
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)
    # To crop the Image present in the Rectangle window
    cropped = frame[bottom_right_y:top_left_y, top_left_x:bottom_right_x]
    img2 = cropped  # Cropped image stored in variable img2
    frame = cv2.flip(frame, 1)  # To flip camera
    orb = cv2.ORB_create(nfeatures=1500,) # Create ORB object named orb
    kp1, des1 = orb.detectAndCompute(img1, None)  # identify the key feature and description of template image
    kp2, des2 = orb.detectAndCompute(img2, None)  # identify the key feature and description of cropped  image
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    n_match = len(matches) # To find number of matches
    matches = sorted(matches, key=lambda x: x.distance)
    matching_results = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
    threshold = 290  # Adjust the threshold value here
    output_string = "Matches=" + str(n_match)

    if(n_match>threshold):
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0,0,255), 3)
        cv2.putText(frame, 'Object Detected', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 3)
        cv2.putText(frame, output_string, (100, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0,), 3)
    else:
        cv2.putText(frame, output_string, (100, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, ), 3)

    #cv2.imshow("Image Template",img1) # To display Template Image

    cv2.imshow("Video Input", frame)
    cv2.imshow("Resulting Matches", matching_results)
    if cv2.waitKey(1)==13:
        break;

cap.release()
cv2.destroyAllWindows()
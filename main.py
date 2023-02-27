import cv2
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector

# resolution = (640, 480)
resolution = (1280,720)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# cap.set(cv2.cv.CV_CAP_PROP_FPS, 60)
cap.set(3, resolution[0])
cap.set(4, resolution[1])

prev_frame_time = 0
new_frame_time = 0

detector = HandDetector(detectionCon=0.8, maxHands=1)

points = np.empty((2,0), np.int32)
imgcanvas = np.zeros((resolution[1],resolution[0],3), np.uint8)
xp,yp=0,0
while True:
    sucess, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
  
    fps = int(fps)
  
    fps = "FPS: "+str(fps)
    # print("FPS: ", fps)
    # putting the FPS count on the frame
    cv2.putText(img, fps, (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

    if hands:
        lmlist = hands[0]["lmList"]
        point_index = lmlist[8][0:2]
        point_middle = lmlist[12][0:2]

        fingers = detector.fingersUp(hands[0])
        # print("UP: ", fingers)
        

        if fingers[1] and fingers[2]:
            xp, yp = 0,0
            cv2.rectangle(img, (point_index[0], point_index[1]), (point_middle[0], point_middle[1]), (255,255,255), cv2.FILLED)
            cv2.rectangle(imgcanvas, (point_index[0], point_index[1]), (point_middle[0], point_middle[1]), (0,0,0), 20, cv2.FILLED)
            # print("Erase mode...")

        if fingers[1] and fingers[2]==False:
            if xp==0 and yp==0:
                xp = point_index[0]
                yp = point_index[1]
            cv2.circle(img, point_index, 10, (0,0,255), cv2.FILLED)
            cv2.line(imgcanvas, (xp,yp), (point_index[0],point_index[1]), (255,0,0), 10)

            xp, yp = point_index[0], point_index[1]

        if fingers[1]==False and fingers[2]==False:
            xp, yp = 0,0

    imgGray = cv2.cvtColor(imgcanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgcanvas)

    # img = cv2.addWeighted(img, 0.5, imgcanvas, 0.5, 0)

    cv2.imshow("Video", img)
    cv2.imshow("Canvas", imgInv)
    cv2.waitKey(1)
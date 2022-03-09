import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)

points = np.empty((2,0), np.int32)

while True:
    sucess, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmlist = hands[0]["lmList"]
        point_index = lmlist[8][0:2]
        points = np.append(points, point_index).reshape((-1,2))
        cv2.circle(img, point_index, 20, (200,0,0), cv2.FILLED)

        # print(points)
    cv2.polylines(img, [points], False, (200,0,0), 20)
        # cv2.fillConvexPoly(img, points, (255,0,0))




    cv2.imshow("Video", img)
    cv2.waitKey(1)
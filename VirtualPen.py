import model.hand as hand
import mediapipe as mp
import cv2
import time
import numpy as np


brushColor = (0, 255, 0)
brushThickNess = 25
eraserThickNess = 50
cameraid=0
width=640
height=480
xp, yp = 0, 0
currentTime = 0
perviousTime = 0

camera = cv2.VideoCapture(cameraid)
imgCanvas = np.zeros((height, width, 3), np.uint8)
detection = hand.HandDetection(trackCon=0.85)

while True:
        success, frame = camera.read()
        frame = cv2.flip(frame, 1)
        frame = detection.findHand(frame)
        landmarks = detection.findPosition(frame, draw=False)
        if len(landmarks) != 0:
            x1, y1 = landmarks[8][1:]
            x2, y2 = landmarks[12][1:]
        # detect up fingers

        fingers = detection.fingersUp()
        if len(fingers) != 0:
            if (fingers[1] and fingers[2]) and ((fingers[3] and fingers[4]) == False):
                xp , yp = 0 , 0 
                cv2.rectangle(frame, (x1, y1 - 15), (x2, y2 + 15),
                              (255, 0, 255), cv2.FILLED)

            if fingers[1] and fingers[2] == False:
                cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                if len(landmarks) != 0:
                    brushColor = (0, 255, 0)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    cv2.line(frame, (xp, yp), (x1, y1),
                             brushColor, brushThickNess)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1),
                             brushColor, brushThickNess)
                    xp, yp = x1, y1
            if fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, "Eraser" , (50 , 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if len(landmarks) != 0:
                    brushColor = (0, 0, 0)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    cv2.line(frame, (xp, yp), (x1, y1),
                             brushColor, eraserThickNess)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1),
                             brushColor, eraserThickNess)
                    xp, yp = x1, y1
        # drawing

        currentTime = time.time()
        fps = 1 / (currentTime - perviousTime)
        perviousTime = currentTime
        cv2.putText(frame, "FPS: " + str(int(fps)), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame ,imgInv)
        frame = cv2.bitwise_or(frame ,imgCanvas)
        cv2.imshow('frame', frame)
        # cv2.imshow('canvas', imgCanvas)
        cv2.waitKey(1)

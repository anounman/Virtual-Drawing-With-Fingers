import mediapipe as mp
import cv2
import time

class HandDetection:
        def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
                self.mode = mode
                self.maxHands = maxHands
                self.detectionCon = detectionCon
                self.trackCon = trackCon
                self.mpHands = mp.solutions.hands
                self.hands = self.mpHands.Hands()
                self.mpDraw = mp.solutions.drawing_utils
                self.tipIds = [4, 8, 12, 16, 20]

        def findHand(self , frame , draw=True):
            RGBimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.result = self.hands.process(RGBimg)
            if self.result.multi_hand_landmarks:
                for hand in self.result.multi_hand_landmarks:
                        if draw:
                            self.mpDraw.draw_landmarks(frame, hand, self.mpHands.HAND_CONNECTIONS)
            return frame
        def findPosition(self , frame , handCount=0 , draw=True):
            self.landmarks = []
            if self.result.multi_hand_landmarks:
                myHand = self.result.multi_hand_landmarks[handCount]
                for id , landmark in enumerate(myHand.landmark):
                    h , w , c = frame.shape
                    cx , cy = int(landmark.x * w ), int(landmark.y * h)
                    self.landmarks.append([id , cx , cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            return self.landmarks
        def fingersUp(self):
            fingers = []
            # Thumb
            if len(self.landmarks) != 0:    
                if self.landmarks[self.tipIds[0]][1] < self.landmarks[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
                # Fingers
                for id in range(1, 5):
                    if self.landmarks[self.tipIds[id]][2] < self.landmarks[self.tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
        
            return fingers

def main(cameraid=0 , isBig=False , width=640 , height=480):
        camera = cv2.VideoCapture(cameraid)
        detection = HandDetection()
        currentTime = 0
        perviousTime = 0
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        while True:
            success, frame = camera.read()
            frame = detection.findHand(frame);
            landmarks =  detection.findPosition(frame)
            fingers = detection.fingersUp()
            # if len(landmarks) != 0:
            #     print(landmarks[4])
            currentTime = time.time()
            fps = 1 / (currentTime - perviousTime)
            perviousTime = currentTime
            cv2.putText(frame, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.flip(frame, 1)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
    
if __name__ == '__main__':
    main(cameraid=0)

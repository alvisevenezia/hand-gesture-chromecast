import numpy as np
import cv2
import signal
import sys
import mediapipe as mp
import time

FRAMERATE = 30
last_time = 0


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    cv2.destroyAllWindows()
    sys.exit(0)

cap = cv2.VideoCapture(0)
map_hand = mp.solutions.hands
hands = map_hand.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

print("running...")

signal.signal(signal.SIGINT, signal_handler)

data_file = open("./training_data/training_data_left-right.csv", "a")

#open a box with the webcan feed and the hand detection
while True:

    if time.time() - last_time < 1/FRAMERATE:
        continue

    last_time = time.time()


    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = hands.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 8:
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if id == 4:
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 255), cv2.FILLED)

            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, map_hand.HAND_CONNECTIONS)

        
    #get pressed key
    key = cv2.waitKey(1) & 0xFF

    if key >= 48 and key <= 57:

        data_to_write = [str(key-48)]

        if results.multi_hand_landmarks:

            for landmark in results.multi_hand_landmarks:
                for lm in landmark.landmark:
                    data_to_write.append(lm.x)
                    data_to_write.append(lm.y)
                    data_to_write.append(lm.z)

            for data in data_to_write:
                data_file.write(str(data) + ";")

            data_file.write("\n")
        


    cv2.imshow('Hand Gesture', frame)
    




cap.release()
cv2.destroyAllWindows()


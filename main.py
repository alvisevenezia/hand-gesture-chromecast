import numpy as np
import cv2
import signal
import sys
import mediapipe as mp
import time
import keras

FRAMERATE = 30
last_time = 0

starting_coordonate_ok = (0, 0)
last_frame_ok = False


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

labels = ["closed","ok","rock","open"]

signal.signal(signal.SIGINT, signal_handler)

#improt keras model 
model = keras.models.load_model("./saved_models/trained/model.keras")

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
        
        data_for_model = []

        for id, lm in enumerate(hand_landmarks.landmark):

            data_for_model.append(lm.x)
            data_for_model.append(lm.y)
            data_for_model.append(lm.z)

        data_for_model = np.array(data_for_model).reshape(1, 63, 1)
        prediction = model.predict(data_for_model,verbose=0)
        prediction_label = labels[np.argmax(prediction)]

        print(prediction_label)

        cv2.putText(frame, prediction_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if labels[np.argmax(prediction)] == "ok":

            if last_frame_ok:
                distance_in_x = starting_coordonate_ok[0] - results.multi_hand_landmarks[0].landmark[8].x

                #draw a line between the two points

                color = (0, 0, 255)

                cv2.line(frame, (int(starting_coordonate_ok[0]*w),int(h/2)), (int(results.multi_hand_landmarks[0].landmark[8].x*w),int(h/2)), color, 2)

            else:  
                starting_coordonate_ok = (results.multi_hand_landmarks[0].landmark[8].x, results.multi_hand_landmarks[0].landmark[8].y)
                last_frame_ok = True
                

        else:
            last_frame_ok = False
        

    cv2.imshow('Hand Gesture', frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break



cap.release()
cv2.destroyAllWindows()

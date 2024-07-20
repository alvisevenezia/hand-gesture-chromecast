import numpy as np
import cv2
import signal
import sys
import mediapipe as mp
import time

import json
import uuid 
import protobuf_to_dict
import copy

FRAMERATE = 300
last_time = 0


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

cap = cv2.VideoCapture(0)
map_hand = mp.solutions.hands

hands = map_hand.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

print("running...")

#extract data info from file 'data.json'
with open('./training_data/data_infos.json') as f:
    data = json.load(f)

print(data)

nbr_of_data = data['nbrOfImages']
folder = data['folder']

#open a box with the webcan feed and the hand detection
while True:

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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

    frame_to_save = copy.deepcopy(frame)

    if results.multi_hand_landmarks:

        hands_box = []
        landmark_data = []

        for hand_landmarks in results.multi_hand_landmarks:

            min_x = 1
            min_y = 1
            max_x = 0
            max_y = 0

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                landmark_data.append([id, cx, cy])
                min_x = min(min_x, lm.x)
                min_y = min(min_y, lm.y)
                max_x = max(max_x, lm.x)
                max_y = max(max_y, lm.y)
                if id == 8:
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if id == 4:
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 255), cv2.FILLED)

            hands_box.append([min_x, min_y, max_x, max_y])

            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, map_hand.HAND_CONNECTIONS)
            

        #generate new uuid
        new_uuid = str(uuid.uuid4())

        #save the image
        cv2.imwrite(f"./training_data/{folder}{new_uuid}.jpg", frame_to_save)

        #save the landmarks, the box coordinate and the number of hand detected in a json file
        with open(f"./training_data/{folder}{new_uuid}.json", 'w') as f:

            json.dump({'landmarks': landmark_data, 'box': hands_box, 'nbrOfHands': len(results.multi_hand_landmarks)}, f)

        #update the data file
        data['images'][data['nbrOfImages']] = {"name": f"{new_uuid}.jpg","label":f"{new_uuid}.json"}
        data['nbrOfImages'] += 1

    cv2.imshow('Hand Recognition', frame)
    

#close the file
with open('./training_data/data_infos.json', 'w') as f:
    json.dump(data, f)


cap.release()
cv2.destroyAllWindows()


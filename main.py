import numpy as np
import cv2
import signal
import sys
import mediapipe as mp
import time
import keras
import sys
from time import sleep

import pychromecast

FRAMERATE = 30
CAST_NAME = "Chambre antoine"
last_time = 0

starting_coordonate_ok = (0, 0)
last_frame_ok = False

starting_coordonate_sound = (0, 0)
last_frame_sound = False
last_delta_sound = 0
SOUND_TRESHOLD = 0.005
SOUND_VARIATION = 0.1

color_green = '\033[92m'
color_red = '\033[91m'
color_end = '\033[0m'
color_blue = '\033[94m'
color_yellow = '\033[93m'

sound_variation_file = open("sound_variation.csv", "w")

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    cv2.destroyAllWindows()
    sys.exit(0)

chromecasts, browser = pychromecast.get_listed_chromecasts(
    friendly_names=[CAST_NAME]
)
if not chromecasts:
    print(color_red+ f'No chromecast with name "{CAST_NAME}" discovered'+color_end)
    sys.exit(1)

cast = chromecasts[0]
cast.wait()

print(color_green+ f'Chromecast "{CAST_NAME}" discovered'+color_end)

cap = cv2.VideoCapture(0)
map_hand = mp.solutions.hands
hands = map_hand.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
    )

print("running...")

labels_pattern = ["closed","time","sound","open"]
labels_hand = ["right","left"]

signal.signal(signal.SIGINT, signal_handler)

#improt keras model 
model_pattern_recognition = keras.models.load_model("./saved_models/trained/model-pattern.keras")
model_hand_recognition = keras.models.load_model("./saved_models/trained/model-hand.keras")
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

        prediction_hand = model_hand_recognition.predict(data_for_model,verbose=0)
        prediction_label_hand = labels_hand[np.argmax(prediction_hand)]

        if labels_hand[np.argmax(prediction_hand)] == "left":
            continue

        prediction_pattern = model_pattern_recognition.predict(data_for_model,verbose=0)
        prediction_label_pattern = labels_pattern[np.argmax(prediction_pattern)]

        cv2.putText(frame, prediction_label_pattern, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if labels_pattern[np.argmax(prediction_pattern)] == "time":

            last_frame_sound = False

            if last_frame_ok:
                distance_in_x = starting_coordonate_ok[0] - results.multi_hand_landmarks[0].landmark[4].x

                color_to_use = color_green if distance_in_x > 0 else color_red

                print(color_blue + "Time : " + color_end + color_to_use + str(distance_in_x) + color_end)
                
            else:  
                starting_coordonate_ok = (results.multi_hand_landmarks[0].landmark[4].x, results.multi_hand_landmarks[0].landmark[4].y)
                last_frame_ok = True

        if labels_pattern[np.argmax(prediction_pattern)] == "sound":
                
            last_frame_ok = False

            if last_frame_sound:
                distance_in_x = starting_coordonate_sound[0] - results.multi_hand_landmarks[0].landmark[4].x
                
                color_to_use = color_green if distance_in_x > 0 else color_red

                curent_delta = distance_in_x - last_delta_sound

                if abs(curent_delta) > SOUND_TRESHOLD:
                    sound_variation_file.write(f"{distance_in_x},{curent_delta}")
                else:
                    sound_variation_file.write(f"{distance_in_x},{curent_delta}\n")
                if curent_delta > SOUND_TRESHOLD:
                    cast.volume_down(SOUND_VARIATION)
                    last_delta_sound = distance_in_x
                    sound_variation_file.write(",down\n")


                elif curent_delta < -SOUND_TRESHOLD:
                    cast.volume_up(SOUND_VARIATION)
                    last_delta_sound = distance_in_x
                    sound_variation_file.write(",up\n")


            else:
                starting_coordonate_sound = (results.multi_hand_landmarks[0].landmark[4].x, results.multi_hand_landmarks[0].landmark[4].y)
                last_frame_sound = True

    else:
        last_frame_ok = False
        last_frame_sound = False

sound_variation_file.close()
cap.release()
cv2.destroyAllWindows()

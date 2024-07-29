#process iamge in folder ./data_to_process in the same way as in create_data.py and save it in folder "./training_data/img"

import mediapipe as mp
import cv2

import json
import os
import uuid
from alive_progress import alive_bar

def main():

    #get current data info 
    with open('./training_data/data_infos.json') as f:
        data = json.load(f)

    folder = data['folder']

    #process image in folder ./data_to_process

    map_hand = mp.solutions.hands
    hands = map_hand.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=True
    )

    for file in os.listdir("./data_to_process"):
        if file.endswith(".jpg"):
            img = cv2.imread(f"./data_to_process/{file}")

            print(f"processing {file}")

            image = cv2.flip(cv2.imread(f"./data_to_process/{file}"), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            landmark_data = [0]*42
            hands_box = [0]*4
            confidence = [1]*21
            nbr_of_data_hands = 0

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    min_x = 1
                    min_y = 1
                    max_x = 0
                    max_y = 0

                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = img.shape

                        landmark_data[id*2] = lm.x
                        landmark_data[id*2 + 1] = lm.y
                            
                        min_x = min(min_x, lm.x)
                        min_y = min(min_y, lm.y)
                        max_x = max(max_x, lm.x)
                        max_y = max(max_y, lm.y)

                    hands_box = [min_x, min_y, max_x, max_y]
                    confidence = [results.multi_handedness[0].classification[0].score]*21
                    nbr_of_data_hands = 1
                        

        #generate new uuid
        new_uuid = str(uuid.uuid4())

        #save the image
        cv2.imwrite(f"./training_data/{folder}{new_uuid}.jpg", img)

        #save the landmarks, the box coordinate, the confidence and the number of hand detected in a json file
        with open(f"./training_data/{folder}{new_uuid}.json", 'w') as f:
            json.dump({"landmarks": landmark_data, "box": hands_box, "confidence": confidence, "nbrOfHands": nbr_of_data_hands}, f)
        
        
    #update the data file
    data['images'][data['nbrOfImages']] = {"name": f"{new_uuid}.jpg","label":f"{new_uuid}.json"}
    data['nbrOfImages'] += 1

    json.dump(data, open('./training_data/data_infos.json', 'w'))

    f.close()

if __name__ == "__main__":
    main()
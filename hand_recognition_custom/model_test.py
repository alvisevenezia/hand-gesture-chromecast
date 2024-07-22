import torch
import cv2

from train_model import HandGestureModel

from torch.utils.data import *

def predict(img,model):
    with torch.no_grad():
        keypoints, confidence = model(img)
    keypoints = keypoints.view(-1, 2)  # Reshape pour obtenir les paires de (x, y)
    confidence = confidence.view(-1)   # Reshape pour obtenir les valeurs de confiance
    return keypoints.cpu().numpy(), confidence.cpu().numpy()

def main():
    model = HandGestureModel()
    model.load_state_dict(torch.load("./models/hand_gesture_model.pth"))
    model.eval()
    with torch.no_grad():
        #draw keypoints
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            scale_x, scale_y = w/128, h/128

            input_frame = cv2.resize(frame, (128,128))
            #input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
            input_frame = torch.tensor(input_frame, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)

            
            outputs = predict(input_frame, model)
            keypoints = outputs[0].reshape(-1)
            confidence = outputs[1]
            
            min_x = 1
            min_y = 1
            max_x = 0
            max_y = 0

            for i in range(21):

                min_x = min(min_x, keypoints[i*2])
                min_y = min(min_y, keypoints[i*2 + 1])
                max_x = max(max_x, keypoints[i*2])
                max_y = max(max_y, keypoints[i*2 + 1])

            frame = cv2.rectangle(frame, (int(min_x*scale_x), int(min_y*scale_y)), (int(max_x*scale_x), int(max_y*scale_y)), (0, 255, 0), 2)
            frame = cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 255, 0), 2)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()
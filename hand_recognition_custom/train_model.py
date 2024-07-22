import torch 
import json
import io
import cv2 

from torch.utils.data import *
from torchvision import transforms
import torch.nn as nn

class HandGestureDataset(Dataset):
    def __init__(self, path, label_type="box", transform=None):
        self.path = path
        self.label_type = label_type
        self.transform = transform

        with open(f"{path}/data_infos.json", 'r') as f:
            self.data_infos = json.load(f)

        self.data_folder = self.data_infos["folder"]

        data = []

        for i in range(self.data_infos["nbrOfImages"]):

            img_path = self.data_infos["images"][str(i)]["name"]
            label_path = self.data_infos["images"][str(i)]["label"]

            data.append((img_path,label_path))

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path, label_path = self.data[idx]

        img = cv2.imread(f"{self.path}/{self.data_folder}{img_path}")
        label_json = json.load(open(f"{self.path}/{self.data_folder}{label_path}"))

        label = []

        if self.transform:
            img = self.transform(img)

        if self.label_type == "box":
            label.extend(label_json["box"])

        elif self.label_type == "landmarks":
            label.extend(label_json["landmarks"])

            
        elif self.label_type == "all":
            label.extend(label_json["box"])
            label.extend(label_json["landmarks"])

        label = torch.tensor(label, dtype=torch.float32)
        confidence = torch.tensor(label_json["confidence"], dtype=torch.float32)
        return img, label, confidence
    


class HandGestureModel(nn.Module):
    def __init__(self, num_keypoints=21):
        super(HandGestureModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.keypoints_layer = nn.Sequential(
            nn.Linear(256, num_keypoints*2),
            nn.Sigmoid()
        )
        self.confidence_layer = nn.Sequential(
            nn.Linear(256, num_keypoints),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        keypoints = self.keypoints_layer(x)
        confidence = torch.sigmoid(self.confidence_layer(x))  # Sigmoid pour obtenir des valeurs entre 0 et 1
        return keypoints, confidence
    
class KeypointConfidenceLoss(nn.Module):
    def __init__(self):
        super(KeypointConfidenceLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, keypoints_pred, keypoints_target, confidence_pred, confidence_target):
        keypoints_loss = self.mse_loss(keypoints_pred, keypoints_target)
        confidence_loss = self.bce_loss(confidence_pred, confidence_target)
        return keypoints_loss + confidence_loss
            

def transform_img_gretscale(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img = transform(img)
    return img

def main():

    #load data

    #convert img ot greyscale
    dataset = HandGestureDataset(path="./training_data", label_type="landmarks", transform=transform_img_gretscale)

    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #hyperparameters
    num_epochs = 1
    num_classes = 21
    learning_rate = 0.001
    batch_size = 128

    #split and load dataset

    test_size = int(len(dataset)*0.2)
    train_set, test_set = random_split(dataset, [len(dataset)-test_size,test_size])
    train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    #build model

    model = HandGestureModel().to(device)

    try:
        model.load_state_dict(torch.load("./models/hand_gesture_model.pth"))
        print("Model loaded")

    except:
        print("Model not found, training new model")

    #train model
    criterion = KeypointConfidenceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        avg_loss = 0.0

        for id, (images, keypoints,confidence) in enumerate(train_loader):
            images, keypoints = images.to(device), keypoints.to(device)

            confidence_target = torch.tensor(confidence, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            outputs, confidences = model(images)
            loss = criterion(outputs, keypoints, confidences, confidence_target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            print_progress(epoch, num_epochs, loss,id+1,len(train_loader),avg_loss/(id+1))

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        

    #save model

    torch.save(model.state_dict(), "./models/hand_gesture_model.pth")

    check_accuracy(test_loader, model, device,criterion)

def print_progress(epoch, num_epochs, loss,batch,batc_size,avg_loss):
    print(f"Epoch [{epoch}/{num_epochs}], Current Batch [{batch}/{batc_size}], Loss: {loss.item()}, AVG Loss : {avg_loss}",end="\r" if batch != batc_size else "\n")


def check_accuracy(val_dataloader, model,device,criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, keypoints, confidence in val_dataloader:
            images, keypoints = images.to(device), keypoints.to(device)
            confidence_target = confidence.to(device)
            outputs, confidences = model(images)
            loss = criterion(outputs, keypoints, confidences, confidence_target)
            val_loss += loss.item() * images.size(0)


    val_loss /= len(val_dataloader.dataset)
    print(f"Validation Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
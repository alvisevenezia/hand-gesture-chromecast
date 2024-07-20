import torch 
import json
import io
import cv2 

from torch.utils.data import *
from torchvision import transforms
import torch.nn as nn

from alive_progress import alive_bar


class HandGestureDataset(Dataset):
    def __init__(self, path, label_type="box", transform=None):
        self.path = path
        self.label_type = label_type
        self.transform = transform

        with open(f"{path}/data_infos.json", 'r') as f:
            self.data_infos = json.load(f)

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

        img = cv2.imread(f"{self.path}/{img_path}")
        label_json = json.load(open(f"{self.path}/{label_path}"))

        label = []

        if self.transform:
            img = self.transform(img)

        if self.label_type == "box":
            for box in label_json["box"]:
                for i in range(4):
                    label.append(box[i])

        elif self.label_type == "landmarks":
            for landmark in label_json["landmarks"]:
                for i in range(3):
                    label.append(landmark[i])

            label = label[:63]
            
        elif self.label_type == "all":
            for box in label_json["box"]:
                for i in range(4):
                    label.append(box[i])

            label = label[:63]

            for landmark in label_json["landmarks"]:
                for i in range(3):
                    label.append(landmark[i])

        label = torch.tensor(label, dtype=torch.float32)

        return img, label
    


class HandGestureModel(nn.Module):
    def __init__(self,in_channels=1, num_classes=63):
        super(HandGestureModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1)
        self.fc1 = nn.Linear(32*32*64, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

    def forward(self, x):
        
        
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
    
            

def transform_img_gretscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    device = torch.device('cpu')

    #hyperparameters
    num_epochs = 1
    num_classes = 63
    learning_rate = 0.001
    batch_size = 64

    #split and load dataset

    test_size = int(len(dataset)*0.2)
    train_set, test_set = random_split(dataset, [len(dataset)-test_size,test_size])
    train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    #build model
    model = HandGestureModel().to(device)

    #train model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        avg_loss = 0

        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            #backward pass
            optimizer.zero_grad()
            loss.backward()

            #optimize
            optimizer.step()
            #print progress

            avg_loss += loss
            print_progress(epoch, num_epochs, loss,i+1,len(train_loader),avg_loss/(i+1))

    #save model

    torch.save(model.state_dict(), "./models/hand_gesture_model.pth")

    check_accuracy(test_loader, model, device)

def print_progress(epoch, num_epochs, loss,batch,batc_size,avg_loss):
    print(f"Epoch [{epoch}/{num_epochs}], Current Batch [{batch}/{batc_size}], Loss: {loss.item()}, AVG Loss : {avg_loss}",end="\r" if batch != batc_size else "\n")


def check_accuracy(loader, model,device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:

            print(x.shape)
            print(y.shape)

            x = x.to(device)
            y = y.to(device)

            scores = model(x)

            print(scores.shape)
            
            for i in range(scores.shape[0]):
                predictions = scores[i]
                labels = y[i]

                print(predictions)
                print(labels)

                if torch.all(predictions.eq(labels)):
                    num_correct += 1

            num_samples += predictions.size(0)

            

        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()

    return float(num_correct)/float(num_samples)


if __name__ == "__main__":
    main()
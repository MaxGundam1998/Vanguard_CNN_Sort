import os
import torch 
import torch.nn as nn
from torch.optim import Adam
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Variable
import glob
import pathlib


#Check if cuda is used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("The device is", device)

#Transform
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

#Test and Train data directories
train_path = 'CardData/NationDetection/card_train/card_train'
test_path = 'CardData/NationDetection/card_train/card_train'

#Load the Data
train_loader = DataLoader(torchvision.datasets.ImageFolder(train_path, transform=transformer), batch_size=32, shuffle=True)
test_loader = DataLoader(torchvision.datasets.ImageFolder(test_path, transform=transformer), batch_size=32, shuffle=False)

class ConvNet(nn.Module):
    def __init__(self, num_classes = 6):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        #Shape = (256, 12, 150, 150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        #Shape = (256, 12, 150, 150)
        self.relu1 = nn.ReLU()
        #Shape = (256, 12, 150, 150)

        self.pool = nn.MaxPool2d(kernel_size=2)
        #Reduce Image size by factor 2
        #Shape = (256, 12, 75, 75)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        #Shape = (256, 20, 75, 75)
        self.relu2 = nn.ReLU()
        #Shape = (256, 20, 75, 75)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        #Shape = (256, 32, 75, 75)
        self.bn3 = nn.BatchNorm2d(num_features = 32)
        #Shape = (256, 32, 75, 75)
        self.relu3=nn.ReLU()
        #Shape = (256, 32, 75, 75)

        self.fc = nn.Linear(in_features=32*75*75, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)
            
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        #Above output will be in matrix form

        output = output.view(-1, 32*75*75)

        output = self.fc(output)

        return output
    
#Retrieve classes
root=pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

model = ConvNet(num_classes=6).to(device)

optimizer = Adam(model.parameters(), lr = 0.001, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

num_epochs = 10

train_count = len(glob.glob(train_path+'/**/*.jpg'))
test_count = len(glob.glob(test_path+'/**/*.jpg'))

print(train_count, test_count)

best_accuracy = 0.0
best_model_path = 'best_overdresspoint.model'

for epoch in range(num_epochs):

    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i , (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data*images.size(0)
            _,prediction = torch.max(outputs.data, 1)

            train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss/train_count

    model.eval()

    test_accuracy = 0.0

    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _,prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy/test_count

    print('Epoch: ' + str(epoch) + ' Train Loss: '+ str(int(train_loss)) + ' Train Accuracy: ' + str(train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

    #Save model

    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_model_path')
        best_accuracy = test_accuracy
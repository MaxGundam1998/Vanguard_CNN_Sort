import torch
import torch.nn as nn
import glob
import os
import shutil
from PIL import Image
from torchvision import transforms

#Testing if this is correct.


#Check if cuda is used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("The device is", device)

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
    
model_path = 'best_overdresspoint.model'
model = ConvNet()
model.load_state_dict(torch.load(model_path))
model.eval()

image_dir = 'CardData/NationSort/NationBundle/'
image_paths = glob.glob(image_dir + '*.jfif')

#Transform
image_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

images = []
for image_path in image_paths:
    image = Image.open(image_path)
    image = image_transforms(image)
    images.append(image)
    
#Stack images into a single tensor
images = torch.stack(images)

#Perform inference
with torch.no_grad():
    outputs = model(images)

# Get predicted class labels
_, predicted = torch.max(outputs, 1)
predicted_classes = predicted.tolist()

class_names = ['Brandt_Gate', 'Dark_States', 'Dragon_Empire', 'Keter_Sanctuary', 'Lyrical', 'Stoichea']

predicted_class_names = [class_names[label] for label in predicted_classes]

output_folders = [
    "CardData/NationSort/SortBrandt",
    "CardData/NationSort/SortDark",
    "CardData/NationSort/SortDragon",
    "CardData/NationSort/SortKeter",
    "CardData/NationSort/SortLyrical",
    "CardData/NationSort/SortStoichea"
]

for i in range(len(image_paths)):

    temp = 0
    image_path = image_paths[i]

    predicted_class = predicted_class_names[i]

    if predicted_class == "Brandt_Gate":
        temp = 0

    elif predicted_class == "Dark_States":
        temp = 1

    elif predicted_class == "Dragon_Empire":
        temp = 2
    
    elif predicted_class == "Keter_Sanctuary":
        temp = 3
    
    elif predicted_class == "Lyrical":
        temp = 4

    elif predicted_class == "Stoichea":
        temp = 5

    output_folder = output_folders[temp]
    filename = os.path.basename(image_path)
    destination_path = os.path.join(output_folder, filename)
    os.makedirs(output_folder, exist_ok=True)
    shutil.copy(image_path, destination_path)

    print(f"Image {i+1}: {image_path} {predicted_class_names[i]}")

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch import optim

df = pd.read_csv('./labels.csv',sep=',')

import os

class_ids = []
filenames = []

# Start with Test
for file in os.listdir("./traffic_Data/TEST/"):
    if file.endswith(".png"):
        filenames.append("./traffic_Data/TEST/" + file)
        classname = file.split("_")[0].lstrip('0')
        if classname == "":
            classname = "0"
        class_ids.append(classname)

classes = df["ClassId"]

for index, value in classes.items():
    for file in os.listdir(os.path.join("./traffic_Data/DATA/", str(value))):
        if file.endswith(".png"):
            filenames.append(os.path.join("./traffic_Data/DATA/", str(value), file))
            classname = file.split("_")[0].lstrip('0')
            if classname == "":
                classname = "0"
            class_ids.append(classname)

df = pd.DataFrame(list(zip(class_ids, filenames)), columns =['classid', 'filename'])
print(df.head())

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, y, BatchSize, transform):
        super().__init__()
        self.BatchSize = BatchSize
        self.y = y
        self.X = X
        self.transform = transform

    def num_of_batches(self):
        """
        Detect the total number of batches
        """
        return math.floor(len(self.list_IDs) / self.BatchSize)

    def __getitem__(self, idx):
        class_id = self.y[idx]
        img = Image.open(self.X[idx])
        img = self.transform(img)
        return img, torch.tensor(int(class_id))

    def __len__(self):
        return len(self.X)

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

# Shuffle dataframe
df = df.sample(frac=1)

X = df.iloc[:,1]
y = df.iloc[:,0]

transform = transforms.Compose([
                transforms.Resize([256,256]),
                transforms.RandomRotation(20, fill=256),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
                transforms.Normalize([0.5], [0.5])
            ])

test_transform = transforms.Compose([
                              transforms.Resize([256,256]),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

train_ratio = 0.90
validation_ratio = 0.05
test_ratio = 0.05

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, stratify = y, random_state = 0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state = 0)

dataset_stages = ['train', 'val', 'test']

batch_size = 32
image_datasets = {'train' : CustomDataset(X_train.values, y_train.values, batch_size, transform),
                  'val' : CustomDataset(X_val.values, y_val.values, batch_size, test_transform),
                  'test' : CustomDataset(X_test.values, y_test.values, batch_size, test_transform)}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=image_datasets[x].BatchSize,
                                            shuffle=True, num_workers=0)
            for x in dataset_stages}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            num_batches = 0
            outputs = None
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # Loading Bar
                if (phase == 'train'):
                    num_batches += 1
                    percentage_complete = ((num_batches * batch_size) / (dataset_sizes[phase])) * 100
                    percentage_complete = np.clip(percentage_complete, 0, 100)
                    print("{:0.2f}".format(percentage_complete), "% complete", end="\r")

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, h = model(inputs)
                    loss = criterion(outputs.float(), labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # TODO: try removal
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                predicted = torch.max(outputs.data, 1)[1]
                running_correct = (predicted == labels).sum()
                running_corrects += running_correct
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects / dataset_sizes[phase]
            # epoch_acc = sum(epoch_acc) / len(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc.item()))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model

from torchvision import models
from torch.optim import lr_scheduler
from SqueezeNet.squeezenet_model import SqueezeNet

model = SqueezeNet(num_classes=58)
model_ft = models.squeezenet1_1(pretrained=True)
model.features.load_state_dict(model_ft.features.state_dict())

for param in model.parameters():
    param.requires_grad = True
for param in model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model.to(device), criterion, optimizer_ft, exp_lr_scheduler, 24)
torch.save(model.state_dict(), f'./checkpoint/squeezenet_ckpt_traffic_data_2_latents.pth')
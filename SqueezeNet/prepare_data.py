import math
import pandas as pd
import torch
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
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

def gen_data():

    df = pd.read_csv('./SqueezeNet/labels.csv',sep=',')
    batch_size = 1

    class_ids = []
    filenames = []

    # Start with Test
    for file in os.listdir("./SqueezeNet/traffic_Data/TEST/"):
        if file.endswith(".png"):
            filenames.append("./SqueezeNet/traffic_Data/TEST/" + file)
            classname = file.split("_")[0].lstrip('0')
            if classname == "":
                classname = "0"
            class_ids.append(classname)

    classes = df["ClassId"]

    for index, value in classes.items():
        for file in os.listdir(os.path.join("./SqueezeNet/traffic_Data/DATA/", str(value))):
            if file.endswith(".png"):
                filenames.append(os.path.join("./SqueezeNet/traffic_Data/DATA/", str(value), file))
                classname = file.split("_")[0].lstrip('0')
                if classname == "":
                    classname = "0"
                class_ids.append(classname)

    df = pd.DataFrame(list(zip(class_ids, filenames)), columns =['classid', 'filename'])
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


    image_datasets = {'train' : CustomDataset(X_train.values, y_train.values, batch_size, transform),
                      'val' : CustomDataset(X_val.values, y_val.values, batch_size, test_transform),
                      'test' : CustomDataset(X_test.values, y_test.values, batch_size, test_transform)}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=image_datasets[x].BatchSize,
                                                shuffle=True, num_workers=0)
                for x in dataset_stages}

    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    return dataloaders

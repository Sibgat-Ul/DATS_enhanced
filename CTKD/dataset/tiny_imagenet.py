import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import ImageOps, ImageEnhance, ImageDraw, Image
import random
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/tiny_imagenet')

class ImageNet(Dataset):
    def __init__(self, files, labels, encoder, transforms, mode):
        super().__init__()
        self.files = files
        self.labels = labels
        self.encoder = encoder
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        pic = Image.open(self.files[index]).convert('RGB')
        x = self.transforms(pic)

        if self.mode == 'train':
            label = self.labels[index]
            y = self.encoder.transform([label])[0]
            return x, y, index

        elif self.mode == 'val':
            label = self.labels[index]
            y = self.encoder.transform([label])[0]
            return x, y

        elif self.mode == 'test':
            return x, self.files[index]

# class ImageNet(ImageFolder):
#     def __getitem__(self, index):
#         img, target = super().__getitem__(index)
#         return img, target, index

def get_imagenet_train_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        transforms.RandomErasing(p=0.5, scale=(0.06, 0.08), ratio=(1, 3), value=0, inplace=True)
    ])

    return train_transform


def get_imagenet_test_transform(mean, std):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    ])
    return test_transform

def get_tiny_imagenet_dataloaders(
        batch_size,
        val_batch_size,
        num_workers,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
):
    DIR_MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/tiny-imagenet-200/')
    DIR_TRAIN = DIR_MAIN + 'train/'
    DIR_VAL = DIR_MAIN + 'val/'
    DIR_TEST = DIR_MAIN + 'test/'

    # Number of labels - 200
    labels = os.listdir(DIR_TRAIN)

    # Initialize labels encoder
    encoder_labels = LabelEncoder()
    encoder_labels.fit(labels)

    # Train
    files_train = []
    labels_train = []
    for label in labels:
        for filename in os.listdir(DIR_TRAIN + label + '/images/'):
            files_train.append(DIR_TRAIN + label + '/images/' + filename)
            labels_train.append(label)

    # Val
    files_val = []
    labels_val = []
    for filename in os.listdir(DIR_VAL + 'images/'):
        files_val.append(DIR_VAL + 'images/' + filename)

    val_df = pd.read_csv(DIR_VAL + 'val_annotations.txt', sep='\t', names=["File", "Label", "X1", "Y1", "X2", "Y2"],
                         usecols=["File", "Label"])
    for f in files_val:
        l = val_df.loc[val_df['File'] == f[len(DIR_VAL + 'images/'):]]['Label'].values[0]
        labels_val.append(l)

    # Test
    files_test = []
    for filename in os.listdir(DIR_TEST + 'images/'):
        files_test.append(DIR_TEST + 'images/' + filename)
        files_test = sorted(files_test)

    train_transform = get_imagenet_train_transform(mean, std)
    test_transform = get_imagenet_test_transform(mean, std)

    train_set = ImageNet(files=files_train,
                              labels=labels_train,
                              encoder=encoder_labels,
                              transforms=train_transform,
                              mode='train')

    num_data = 200

    train_loader = DataLoader(train_set, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True)

    test_set = ImageNet(files=files_val,
                            labels=labels_val,
                            encoder=encoder_labels,
                            transforms=test_transform,
                            mode='val')

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, num_data
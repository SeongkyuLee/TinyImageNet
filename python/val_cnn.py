#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:39:12 2017

@author: Q
"""

from torch.utils.data import DataLoader
from torchvision import transforms

import torch
from torch.autograd import Variable
from dataset import TrainDataset
from cnnmodel import CNN, CNN2

BATCH_SIZE = 10
MOMENTUM = 0.5



#==============================================================================
# load dataset
#==============================================================================
IMG_EXT = ".JPEG"
TRAIN_IMG_PATH = "../data/train/images/"
VAL_DATA = "../data/train/validation.csv"
MODEL_PATH = "../data/model.pkl"
OUTPUT_PATH = "../data/output.txt"

is_cuda = torch.cuda.is_available()
model = CNN3()
model.load_state_dict(torch.load(MODEL_PATH))
if is_cuda:
    model.cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
#transformations = transforms.Compose([transforms.Scale(32),transforms.ToTensor()])

transformations = transforms.Compose([
            transforms.Scale(56),
            transforms.CenterCrop(56),
            transforms.ToTensor(),
            normalize,
        ])
kwargs = {'num_workers':1, 'pin_memory':True} if is_cuda else {}
#transformations = transforms.Compose([transforms.Scale(32),transforms.ToTensor()])
val_dataset = TrainDataset(VAL_DATA, TRAIN_IMG_PATH, IMG_EXT, transformations)

val_loader = DataLoader(val_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle=False,
                          **kwargs)

# Test the Model
print('Validation start')
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for i, (images, labels) in enumerate(val_loader):
    images = Variable(images)
    if is_cuda:
        images = images.cuda()
        labels = labels.cuda()
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

    if (i+1) % 100 == 0:
        print ('Iter [%d/%d]' 
               %(i+1, len(val_dataset)//BATCH_SIZE))
    
print('Test Accuracy of the model on the %d test images: %d %%' % (len(val_dataset), 100 * correct / total))

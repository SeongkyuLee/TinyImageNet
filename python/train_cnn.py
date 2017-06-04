#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:33:46 2017

@author: Q
"""
from torch.utils.data import DataLoader
from torchvision import transforms

import torch
from torch.autograd import Variable

from dataset import TrainDataset
from cnnmodel import CNN,CNN2,CNN3

BATCH_SIZE = 1
LR = 0.001
MOMENTUM = 0.5
NUM_EPOCHS = 5

#==============================================================================
# load model and dataset
#==============================================================================
IMG_EXT = ".JPEG"
TRAIN_IMG_PATH = "../data/train/images/"
#TRAIN_DATA = "../data/train/train_labels.csv"
TRAIN_DATA = "../data/train/train.csv"
MODEL_PATH = "../data/model.pkl"

is_cuda = torch.cuda.is_available()
model = CNN3()


if is_cuda:
    model.cuda()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

#transformations = transforms.Compose([transforms.Scale(32),transforms.ToTensor()])


transformations = transforms.Compose([
            transforms.RandomSizedCrop(56),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

kwargs = {'num_workers':1, 'pin_memory':True} if is_cuda else {}
train_dataset = TrainDataset(TRAIN_DATA, TRAIN_IMG_PATH, IMG_EXT, transformations)
train_loader = DataLoader(train_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle=True,
                          **kwargs)
#==============================================================================
# make neural net
#==============================================================================

# Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#optimizer = torch.optim.SGD(model.parameters(), lr=LR,momentum=0.01)

print('Start training')
model.train()
# Train the Model
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        
        if is_cuda:
            images = images.cuda()
            labels = labels.cuda()
        
        images = Variable(images)
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        
        labels = labels.view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, NUM_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE, loss.data[0]))
        
    torch.save(model.state_dict(), MODEL_PATH)

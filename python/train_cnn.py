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
from dataset import TrainDataset, save_fig
from model import make_CNN
import sys

def train(model_number, is_train):
    BATCH_SIZE = 100
    LR = 0.001
    NUM_EPOCHS = 20
    
    # load model and dataset
    IMG_EXT = ".JPEG"
    TRAIN_IMG_PATH = "../data/train/images/"
    MODEL_PATH = "../model/CNN" + model_number + "_model.pkl"
    LOSS_FIG_PATH = "../figure/CNN" + model_number + "_loss.jpg"
    LOSS_FIG_TITLE = "CNN" + model_number + " loss"
 
    model = make_CNN(model_number)
    if int(is_train):
        print('Train model only with 40,000 images.')
        TRAIN_DATA = "../data/train/train.csv"        
    else:
        print('Train model with 50,000 images.')
        model.load_state_dict(torch.load(MODEL_PATH))
        TRAIN_DATA = "../data/train/validation.csv"        

    # check whether use cuda or not
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transformations = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
    
    kwargs = {'num_workers':1, 'pin_memory':True} if is_cuda else {}
    train_dataset = TrainDataset(TRAIN_DATA, TRAIN_IMG_PATH, IMG_EXT, transformations)
    train_loader = DataLoader(train_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle=True,
                              **kwargs)

    # Loss and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train the Model    
    print('Start training')
    model.train()
    losses = []
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
            losses.append(loss.data[0])
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                       %(epoch+1, NUM_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE, loss.data[0]))

    print('Save model')
    torch.save(model.state_dict(), MODEL_PATH)    
    save_fig(losses, LOSS_FIG_PATH, LOSS_FIG_TITLE )
    
if __name__ == '__main__':
    train(sys.argv[1], sys.argv[2])

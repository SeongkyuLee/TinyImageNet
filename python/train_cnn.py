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
from vggmodel import make_vgg
from resnetmodel import make_resnet
import pandas as pd
import sys

def train(model_name, model_number, is_train, pretrained):
    BATCH_SIZE = 100
    LR = 0.001
    NUM_EPOCHS = 10
    
    # load model and dataset
    IMG_EXT = ".JPEG"
    TRAIN_IMG_PATH = "../data/train/images/"
    MODEL_PATH = "../model/" + model_name + model_number + "_model.pkl"
    LOSS_PATH = "../figure/" + model_name + model_number + "_loss.csv"
    LOSS_FIG_PATH = "../figure/" + model_name + model_number + "_loss.jpg"
    LOSS_FIG_TITLE = "CNN" + model_name + model_number + " loss"
 
    if model_name == "vgg":
        model = make_vgg(model_number)
    elif model_name == "resnet":
        model = make_resnet(model_number)
    else:
        print('choose valid model among vgg and resnet')

    if int(pretrained):
        print('load pretrained model')
        model.load_state_dict(torch.load(MODEL_PATH))
 
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
								transforms.Scale(64),
								transforms.RandomCrop(56),
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
    df = pd.DataFrame.from_records(losses)
    df.to_csv(LOSS_PATH, index=False) 
    save_fig(losses, LOSS_FIG_PATH, LOSS_FIG_TITLE )
    
if __name__ == '__main__':
    print(sys.argv)
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
    print(sys.argv[4])
    train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

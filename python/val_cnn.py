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
#from cnnmodel import CNN, CNN2, CNN3
from model import CNN1, CNN2, CNN3, CNN4
import sys

def validate(model_name):
    BATCH_SIZE = 100
    models = {'CNN1':CNN1(), 'CNN2':CNN2(), 'CNN3':CNN3(), 'CNN4':CNN4()}
    # load model and dataset

    IMG_EXT = ".JPEG"
    VAL_IMG_PATH = "../data/train/images/"
    VAL_DATA = "../data/train/validation.csv"
    MODEL_PATH = "../model/"+model_name+"_model.pkl"
    
    print('Validate model with 10,000 images.')
    model = models[model_name]
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # check whether use cuda or not
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transformations = transforms.Compose([
                transforms.ToTensor(),
                normalize])
    kwargs = {'num_workers':1, 'pin_memory':True} if is_cuda else {}

    val_dataset = TrainDataset(VAL_DATA, VAL_IMG_PATH, IMG_EXT, transformations)
    val_loader = DataLoader(val_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle=False,
                              **kwargs)
    
    # validate the Model
    print('Validation start')
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(val_loader):
        if is_cuda:
            images = images.cuda()
            labels = labels.cuda()
        images = Variable(images)
        outputs = model(images)
    
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    
        if (i+1) % 10 == 0:
            print ('Iter [%d/%d] Accuracy: %.4f' 
                   %(i+1, len(val_dataset)//BATCH_SIZE, 100 * correct / total))
        
    print('Test Accuracy of the model on the %d test images: %d %%' % (len(val_dataset), 100 * correct / total))

if __name__ == '__main__':
    validate(sys.argv[1])
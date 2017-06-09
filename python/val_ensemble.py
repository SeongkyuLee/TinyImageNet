#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 23:12:21 2017

@author: Q
"""


from torch.utils.data import DataLoader
from torchvision import transforms

import torch
from torch.autograd import Variable
from dataset import TrainDataset, save_fig
from vggmodel import make_vgg
from resnetmodel import make_resnet
import sys
import csv

def validate(model_name, model_number):
    BATCH_SIZE = 10
    
    # load model and dataset

    IMG_EXT = ".JPEG"
    VAL_IMG_PATH = "../data/train/images/"
    VAL_DATA = "../data/train/validation.csv"
    MODEL_PATH1 = "../model/"+ model_name + model_number + "_val_1.pkl"
    MODEL_PATH2 = "../model/"+ model_name + model_number + "_val_2.pkl"   
    ACC_PATH1 = "../figure/" + model_name + model_number + "_accuracy_1.csv"       
    ACC_FIG_PATH1 = "../figure/" + model_name + model_number + "_accuracy_1.jpg"
    ACC_FIG_TITLE1 = model_name + model_number + "model 1 accuracy"   
    ACC_PATH2 = "../figure/" + model_name + model_number + "_accuracy_2.csv"       
    ACC_FIG_PATH2 = "../figure/" + model_name + model_number + "_accuracy_2.jpg"
    ACC_FIG_TITLE2 = model_name + model_number + "model 2 accuracy"
    ACC_ENSEMBLE_PATH = "../figure/" + model_name + model_number + "_ensemble_accuracy.csv"       
    ACC_ENSEMBLE_FIG_PATH = "../figure/" + model_name + model_number + "_ensemble_accuracy.jpg"
    ACC_ENSEMBLE_FIG_TITLE = model_name + model_number + "model ensemble accuracy"

    if model_name == "vgg":
        model1 = make_vgg(model_number)
        model2 = make_vgg(model_number)        
    elif model_name == "resnet":
        model1 = make_resnet(model_number)
        model2 = make_resnet(model_number)        
    else:
        print('choose valid model among vgg and resnet')
        
    print('Validate model with 10,000 images.')
    model1.load_state_dict(torch.load(MODEL_PATH1))
    model2.load_state_dict(torch.load(MODEL_PATH2))    
    
    # check whether use cuda or not
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model1.cuda()
        model2.cuda()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transformations = transforms.Compose([
                transforms.Scale(224),
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
    accuracies = []
    accuracies1 = []
    accuracies2 = []
    model1.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    model2.eval()    
    correct = 0
    correct1 = 0
    correct2 = 0
    total = 0
    for i, (images, labels) in enumerate(val_loader):
        if is_cuda:
            images = images.cuda()
            labels = labels.cuda()
        images = Variable(images)
        outputs1 = model1(images)
        outputs2 = model2(images)    	
        probs1, predicted1 = torch.max(outputs1.data, 1)
        probs2, predicted2 = torch.max(outputs2.data, 1)

        predicted = []
        for j in range(len(probs1)):
            if probs1[j].max() > probs2[j].max():
                prediction = predicted1[j].max()
            else:
                prediction = predicted2[j].max()
            predicted.append(prediction)

        predicted = torch.LongTensor(predicted).cuda()
        total += labels.size(0)
        correct += (predicted == labels).sum()
        correct1 += (predicted1 == labels).sum()
        correct2 += (predicted2 == labels).sum()

        accuracies.append(100 * correct / float(total))
        accuracies1.append(100 * correct1 / float(total))
        accuracies2.append(100 * correct2 / float(total))
    
        if (i+1) % 10 == 0:
            print ('Iter [%d/%d] Accuracy: %.4f' 
                   %(i+1, len(val_dataset)//BATCH_SIZE, 100 * correct / total))
        
    print('Test Accuracy of the model on the %d test images: %d %%' % (len(val_dataset), 100 * correct / total))
    save_fig(accuracies, ACC_ENSEMBLE_FIG_PATH, ACC_ENSEMBLE_FIG_TITLE)
    save_fig(accuracies1, ACC_FIG_PATH1, ACC_FIG_TITLE1)
    save_fig(accuracies2, ACC_FIG_PATH2, ACC_FIG_TITLE2)

    with open(ACC_ENSEMBLE_PATH, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(accuracies)
    with open(ACC_PATH1, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(accuracies1)
    with open(ACC_PATH2, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(accuracies2)
   
if __name__ == '__main__':
    validate(sys.argv[1], sys.argv[2])

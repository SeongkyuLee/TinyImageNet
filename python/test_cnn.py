#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:27:34 2017

@author: Q
"""

import torch
from dataset import TestDataset
from model import make_CNN
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import pandas as pd
import sys

def test(model_number):
    BATCH_SIZE = 100
    IMG_EXT = ".JPEG"
    TEST_IMG_PATH = "../data/test/images/"
    TEST_DATA = "../data/test/test_sample_submission_kaggle.csv"
    MODEL_PATH = "../model/"+model_number+"_model.pkl"
    OUTPUT_PATH = "../result/"+model_number+"_result.csv"
    
    is_cuda = torch.cuda.is_available()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
                transforms.ToTensor(),
                normalize])
    
    kwargs = {'num_workers':1, 'pin_memory':True} if is_cuda else {}
    test_dataset = TestDataset(TEST_DATA, TEST_IMG_PATH, IMG_EXT, transformations)
    
    test_loader = DataLoader(test_dataset,
                             batch_size = BATCH_SIZE,
                             shuffle=False,
                             **kwargs)
    
    model = make_CNN(model_number)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    if is_cuda:
        model.cuda()

    model.eval()
    
    header = list()
    header.append('id')
    class_names = ['class_'+str(x).zfill(3) for x in range(100)]
    
    for i in range(len(class_names)):
        header.append(class_names[i])
    
    results = list()
    
    print('Testing start')
    for i, (images, ids) in enumerate(test_loader):
        if is_cuda:
            images = images.cuda()
        images = Variable(images)
        outputs = model(images) 
        
        for j in range(len(outputs)):
            result = list()
            result.append(ids[j])
            output = list(outputs[j].data)
            for k in range(len(output)):            
                result.append(2 ** output[k]) #log -> probability
                #result.append(round(output[k],2))
            results.append(result)
            
        if (i+1) % 10 == 0:      
            print ('Iter [%d/%d]'%(i+1, len(test_dataset)//BATCH_SIZE))
    
    
    df = pd.DataFrame.from_records(results, columns=header)
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == '__main__':
    test(sys.argv[1])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:48:36 2017

@author: Q
"""
import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, csv_path, img_path, img_ext, transform=None):  
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['id'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
        "Some images referenced in the CSV file were not found"
        
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['id']
        self.y_train = tmp_df['label']

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        target = torch.LongTensor([self.y_train[index].item()])        
        #target = np.array([self.y_train[index]])
        return img, target

    def __len__(self):
        return len(self.X_train.index)
    
class TestDataset(Dataset):
    def __init__(self, csv_path, img_path, img_ext, transform=None):
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['id'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
        "Some images referenced in the CSV file were not found"
        
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['id']
        #self.y_train = tmp_df['label']

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        id = self.X_train[index]
        return img, id

    def __len__(self):
        return len(self.X_train.index)
    
def save_fig(y, path, title):    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title='Example') 
     
    ax.plot(y)
    plt.savefig(path)
    

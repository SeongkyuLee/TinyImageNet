#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:57:10 2017

@author: Q
"""
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16), #same as output depth
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
                nn.Linear(32*14*14, 100)) # hard coding

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out)
        return out
    
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16), #same as output depth
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc = nn.Sequential(
                nn.Linear(32*14*14, 3000),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(3000,1000),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1000,100))# hard coding

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out)
        return out
    
class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), #same as output depth
            nn.ReLU(True),
            nn.MaxPool2d(2,2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2,2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2,2))
        
        self.fc = nn.Sequential(
                nn.Linear(256*7*7, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096,4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096,100))# hard coding

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
    
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out)
        return out

   

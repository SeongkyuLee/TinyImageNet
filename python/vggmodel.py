#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 00:20:50 2017

@author: Q
"""
import torch.nn as nn
import math

class vgg(nn.Module):

    def __init__(self, model_number, cnn_layers):
        super(vgg, self).__init__()
        model_number = int(model_number)
        self.cnn_layers = cnn_layers
        
        if  model_number < 5:
            self.fc_layers = nn.Sequential(
                nn.Linear(128 * 3 * 3, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, 100),
            )

            
        elif 5 <= model_number < 10:
            self.fc_layers = nn.Sequential(
                nn.Linear(128 * 7 * 7, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, 100),
            )
       
        elif 10 <= model_number <12:
            self.fc_layers = nn.Sequential(
                nn.Linear(128 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 100),
            )            
 
        elif 12 <= model_number <13:
            self.fc_layers = nn.Sequential(
                nn.Linear(256 * 3 * 3, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 100),
            ) 
            
        elif 13 <= model_number:
            self.fc_layers = nn.Sequential(
                nn.Linear(128 * 3 * 3, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 100),
            )                                        

        self._initialize_weights()

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
def make_convolutional_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    # nn.Linear(128 * 3 * 3, 1024) for 1~4
    '1': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    '2': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    '3': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    '4': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M'],
    # nn.Linear(128 * 7 * 7, 1024) for 5~9
    '5': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128],
    '6': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128],
    '7': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128],
    '8': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128],
    '9': [16, 16, 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],  
    # nn.Linear(128 * 7 * 7, 4098) for 10~11
    '10': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128],
    '11': [16, 16, 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    # nn.Linear(128 * 3 * 3, 4098) for 12
    '12': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256],
    '13': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    '14': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128],    
}

def make_vgg(model_number):
    cnn_layers = make_convolutional_layers(cfg[model_number],True)
    model = vgg(model_number, cnn_layers)
    return model


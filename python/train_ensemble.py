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
import csv
import sys
def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train(model_name, model_number, model_index, is_validation):
    is_validation = int(is_validation)
    BATCH_SIZE = 100
    LR = 0.001
    NUM_EPOCHS = 10
    
    # load model and dataset
    IMG_EXT = ".JPEG"
    TRAIN_IMG_PATH = "../data/train/images/"
    MODEL_PATH = "../model/" + model_name + model_number + "_test_" + model_index + ".pkl"
    LOSS_PATH = "../figure/" + model_name + model_number + "_test_loss_" + model_index + ".csv"
    LOSS_FIG_PATH = "../figure/" + model_name + model_number + "_test_loss_" + model_index + ".jpg"
    LOSS_FIG_TITLE = "CNN" + model_name + model_number + "test loss"
 
    if model_name == "vgg":
        model = make_vgg(model_number)
    elif model_name == "resnet":
        model = make_resnet(model_number)
    else:
        print('choose valid model among vgg and resnet')

    if(is_validation): 
        print('Train model with 45,000 images.')
        TRAIN_DATA = "../data/train/train.csv"
    else:
        print('Train model with 50,000 images.')
        TRAIN_DATA = "../data/train/train_labels.csv"        
   
    # check whether use cuda or not
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print('I use cuda')
        model.cuda()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if model_name == 'resnet' and (model_number == '18' or model_number == '34'):
        transformations = transforms.Compose([
    								transforms.Scale(256),
    								transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    else:
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
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    # Train the Model    
    print('Start training')
    model.train()
    losses = []
    for epoch in range(NUM_EPOCHS):
        optimizer = exp_lr_scheduler(optimizer, epoch, LR, 5)
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
            if (i+1) % 10 == 0:
                losses.append(loss.data[0])
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                       %(epoch+1, NUM_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE, loss.data[0]))

    print('Save model')
    torch.save(model.state_dict(), MODEL_PATH)    
    save_fig(losses, LOSS_FIG_PATH, LOSS_FIG_TITLE )
    
    with open(LOSS_PATH, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(losses)
    
if __name__ == '__main__':
    train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

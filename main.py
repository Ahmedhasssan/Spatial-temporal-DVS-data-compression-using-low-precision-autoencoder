This repo is created by Ahmed Hasssan 
Arizona State University

import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import dvs
from utils import *
from models import *
from training import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
### Train and Test Data ###
#trainloader, validloader, testloader, train_bb, valid_bb, test_bb = dvs.get_mnist_dvs(root="/home/ahasssan/ahmed/dvs/", batch_size=128)
trainloader, testloader = dvs.get_mnist_dvs(root="/home/ahasssan/ahmed/dvs/", batch_size=64)
#import pdb;pdb.set_trace()
#train_set, val_set = torch.utils.data.random_split(trainloader, [30, 11])
classifier_dataloader = trainloader
autoencoder_train_dataloader = trainloader
autoencoder_test_dataloader = testloader
#import pdb;pdb.set_trace()

########### Main Function #########
### Model parameters to Cuda #######
classifier = CLassifier().to(device)
#classifier = ResNet50(101).to('cuda')
autoencoder = Autoencoder().to(device)
QUautoencoder = QUAutoencoder().to(device)
qautoencoder = QuantAutoencoder().to(device)
#qautoencoder = QAutoencoder().to(device)
#qautoencoder = QAutoencoder()   
### Train Classifier ####
#train_accuracy,test_accuracy =Train_classifier(100, trainloader, testloader, classifier)
#print('Classifier Train Accuracy: Origonal data', train_accuracy)
#print('Classifier Test Accuracy: Original data', test_accuracy)
#torch.save(classifier.state_dict(), 'classifer_NMNIST_256.pth')
### Train and Validation Progress ##
path = "/home/ahasssan/ahmed/classifer_NMNIST_256.pth"
#path = "/home/ahasssan/ahmed/classifer_NCaltech.pth"
#path = "/home/ahasssan/ahmed/classifer_Ncaltech_64.pth"
classifier = CLassifier()
classifier.load_state_dict(torch.load(path))
##accuracy, overall_sparsity,group_sparsity, sparse_group=Train(2, trainloader, testloader, classifier, QUautoencoder)
accuracy, overall_sparsity, Image_difference=Train(30, trainloader, testloader, classifier, QUautoencoder)
print('Encoded dataset: Test Accuracy', accuracy)
print('overall_sparsity of Encoder Part:', overall_sparsity)
print('RMSE between real and decoder image:', Image_difference)
#print('Group sparsity', group_sparsity)
#print('sparse groups', sparse_group)
#print('Final losss value', loss)
#print("========================")
#accuracy = Test(testloader, classifier, autoencoder)
#print('Test Accuracy', accuracy)
#print("========================")
# free CUDA memory
del autoencoder
torch.cuda.empty_cache()

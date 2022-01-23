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
import IPython
import torch.nn.init as init
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional, cast
from torch import Tensor
from collections import OrderedDict 

from modules import *
from PACT import *

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #self.conv1=nn.Conv2d(1, 16, 3, 1)
        #self.relu1=nn.ReLU()
        #self.pool1=nn.MaxPool2d(2, 2)
        #self.conv2=nn.Conv2d(16, 32, 3, 1)
        #self.relu2=nn.ReLU()
        #self.pool2=nn.MaxPool2d(2, 2)
        self.background = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
            )
        #self.linear1=nn.Linear(128*57*42*32, 500)
        #self.relu3=nn.ReLU()
        #self.linear2=nn.Linear(500, 101)
        self.projection = nn.Sequential(
            nn.Linear(432*32, 500),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(300, 101),
            nn.LogSoftmax(),
            )
    def forward(self, x):
        x=self.background(x)
        x=self.projection(x)
        return x

class CLassifier(nn.Module):
    def __init__(self):
        super(CLassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(123008, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
            nn.LogSoftmax(),
            )
    def forward(self, x):
        return self.model(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2,padding=1),
            #nn.AvgPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2,padding=1),
            #nn.AvgPool2d(2,2),
            nn.ReLU()
            )
        self.decoder = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            #nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
            )
    def forward(self, x):
        x = self.encoder(x)
        x = torch.sigmoid(x)
        x = torch.logit(x, eps=0.001)
        x = self.decoder(x)
        return x
        
class QUAutoencoder(nn.Module):
    def __init__(self):
        super(QUAutoencoder,self).__init__()
        self.encoder = nn.Sequential(
            QConv2d(1, 32, 3, stride=1, padding=1, wbit=2, abit=4),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            QConv2d(32, 16, 3, stride=1, padding=1, wbit=2, abit=4),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            )   
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            #nn.Hardtanh()
            )
    def forward(self, x):
        x = self.encoder(x)
        x = torch.sigmoid(x)
        encoded = x
        x = torch.logit(x, eps=0.001)
        x = self.decoder(x)
        return x
        
    def get_group_mp(self):
        self.ch_group=1
        val = torch.Tensor()
        if torch.cuda.is_available():
            val = val.cuda()

        count = 0
        for m in self.modules():
            if isinstance(m, QConv2d):
                kw = m.weight.size(2)
                if kw != 1:
                    if not count in [0]:
                        w_l = m.weight
                        num_group = w_l.size(0) * w_l.size(1) // self.ch_group
                        w_l = w_l.view(w_l.size(0), w_l.size(1) // self.ch_group, self.ch_group, kw, kw)
                        w_l = w_l.contiguous().view((num_group, self.ch_group*kw*kw))

                        g = w_l.abs().mean(dim=1)
                        val = torch.cat((val.view(-1), g.view(-1)))
                    count += 1
        return val  
          
    def get_global_mp_thre(self, ratio):
        grp_val = self.get_group_mp()
        sorted_block_values, indices = torch.sort(grp_val.contiguous().view(-1))
        thre_index = int(grp_val.data.numel() * ratio)
        threshold = sorted_block_values[thre_index]
        return threshold
        
class NewModel(nn.Module):
    def __init__(self, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        #print(self.output_layers)
        self.selected_out = OrderedDict()
        #PRETRAINED MODEL
        self.pretrained = QUAutoencoder()
        self.fhooks = []

        for i,l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.pretrained,l).register_forward_hook(self.forward_hook(l)))
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out 
        
class QAutoencoder(nn.Module):
    def __init__(self):
        super(QAutoencoder, self).__init__()
        self.conv1 = QConv2d(1, 16, 3, stride=2, padding=1, wbit=8, abit=8)
        #self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = QConv2d(16, 32, 3, stride=2, padding=1, wbit=8, abit=8)
        #self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.d_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.d_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(1)
        self.relu4 = nn.ReLU()
        #self.pool = nn.MaxPool2d(2,2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = torch.sigmoid(x)
        x = torch.logit(x, eps=0.001)
        
        x = self.d_conv1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.d_conv2(x)
        x = self.bn4(x)
        x = self.relu4(x)
        #x = torch.sigmoid(x)
        x = torch.hardtanh(x)
        #x = self.pool(x)
        
        return x
        
class denoisingautoencoder(nn.Module):
    def __init__(self):
        super(denoisingautoencoder, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)  
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x) # the latent space representation
        
        # decode
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.sigmoid(self.out(x))
        return x

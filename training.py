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
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import statistics as stat

import os
import dvs
from utils import *
from models import *
from sparsity import *
from modules import *
import seaborn as sns
from resnet import *
from googlenet import *
from ResNet import Bottleneck, ResNet, ResNet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def img_quant(img):
    image = img
    image[image.le(0.25)]=0
    image[image.ge(0.25)*image.le(0.75)]=0.5
    image[image.ge(0.75)*image.le(1)]=1
    return image 


def glasso_global_mp(var, dim=0, thre=0.0):
    if len(var.size()) == 4:
        var = var.contiguous().view((var.size(0), var.size(1) * var.size(2) * var.size(3)))

    a = var.pow(2).sum(dim=dim).pow(1/2)
    b = var.abs().mean(dim=1)

    penalty_groups = a[b<thre]
    return penalty_groups.sum(), penalty_groups.numel()
    
def kl_divergence(p, q):
    '''
    args:
        2 tensors `p` and `q`
    returns:
        kl divergence between the softmax of `p` and `q`
    '''
    p = torch.tensor([p] * len(q)).to(device)
    p = F.softmax(p)
    q = F.softmax(q)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


def sparse_loss(autoencoder, images, children):
    loss = 0
    values = images
    model_children = children
    #import pdb;pdb.set_trace()
    #print(len(model_children))
    #for i in range(2):
    values = F.relu(model_children[0](values))
    #import pdb;pdb.set_trace()
    #print(values.unique().size())
    #print("=====================")
    loss += torch.mean(torch.abs(values))
    return loss
def save_imgs(filename, top, bottom, num_imgs = 1):
    plt.figure(figsize=(20, 5))
    for i in range(num_imgs):
        # display original
        ax = plt.subplot(2, num_imgs, i + 1)
        plt.imshow(top[i].cpu().detach().numpy().reshape(256,256), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, num_imgs, i + 1 + num_imgs)
        #plt.imshow(bottom[i].cpu().detach().numpy().reshape(174,234), cmap='gray')
        plt.imshow(bottom[i].cpu().detach().numpy().reshape(256,256), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(filename)
#### Train Classifier #####
def Train_classifier(epochs, trainloader, testdata, model):
    classifier = model.to(device)
    #classifier=ResNet18().to(device)
    #if device == 'cuda':
    #  net = torch.nn.DataParallel(net)
    #  cudnn.benchmark = True
    distance = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.5)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    time_step=20
    num_epochs=epochs
    classifier_dataloader = trainloader
    #d,l=trainloader
    #import pdb;pdb.set_trace()
    classifier_test_dataloader = testdata
    for epoch in range(num_epochs):
        scheduler.step()
        for data, target in classifier_dataloader:
          ####### For N-Caltech Dataset
          model_input_size = torch.tensor([256, 256])
          data=data.squeeze(2)
          data = torch.nn.functional.interpolate(data, torch.Size(model_input_size))
          data=data.unsqueeze(2)
          ## Three channels ####
          #data=data.repeat(1,1,3,1,1)
          #####################
          #####################################
          #target = int(target/target.max())
          l1=0
          running_loss=0
          for step in range(time_step):
            x = data[:, step]
            x /= x.max()
            #import pdb;pdb.set_trace()
            #print(x.shape)
            target = target.long()
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            output = classifier(x)
            loss = distance(output, target)
            #scheduler.step(loss)
            l1+=1
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
          loss1 = running_loss/l1
          #sample=sample+1
        print('epoch {:d}/{:d}, loss: {:.4f}'.format(epoch+1, num_epochs, loss1), end='\r')

    correct = 0
    classifier = classifier.to('cpu')
    total=0
    acc=0
    a=0
    for data, target in classifier_dataloader:
        ####### For N-Caltech Dataset
        model_input_size = torch.tensor([256, 256])
        data=data.squeeze(2)
        data = torch.nn.functional.interpolate(data, torch.Size(model_input_size))
        data=data.unsqueeze(2)
        ## Three channels ###
        #data=data.repeat(1,1,3,1,1)
        #####################
        #####################################
        #target = int(target/target.max())
        for step in range(time_step):
          x = data[:, step]
          x /= x.max()
          output = classifier(x)
          ###
          _, predicted = torch.max(output.data, 1)
          total += target.size(0)
          correct += (predicted == target).sum().item()
          ##
        a+=1
        acc+=100*correct/total
    print('Classifier train accuracy: {:.1f}%'.format(acc/a))
    
    correct = 0
    total=0
    test_acc=0
    b=0
    for data, target in classifier_test_dataloader:
        ####### For N-Caltech Dataset #########
        model_input_size = torch.tensor([256, 256])
        data=data.squeeze(2)
        data = torch.nn.functional.interpolate(data, torch.Size(model_input_size))
        data=data.unsqueeze(2)
        ## Three Channels #########
        #data=data.repeat(1,1,3,1,1)
        ###########################
        #####################################
        #target = int(target/target.max())
        for step in range(time_step):
          x = data[:, step]
          x /= x.max()
          output = classifier(x)
          ###
          _, predicted = torch.max(output.data, 1)
          total += target.size(0)
          correct += (predicted == target).sum().item()
          ##
        b+=1
        test_acc+=100*correct/total
    print('Classifier Test accuracy: {:.1f}%'.format(test_acc/b))
    return acc/a, test_acc/b
    
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

#### Train Autoencoder Function #####
def Train(epochs, trainloader, testdata, model1, model2):
    results = []
    #classifier = model1.to('cpu')
    classifier = model1
    autoencoder = model2.cuda()
    #autoencoder.summary()
    #import pdb;pdb.set_trace()
    distance = nn.BCELoss().cuda()
    #distance = nn.MSELoss()
    #distance = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    #scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    num_epochs = epochs
    autoencoder_train_dataloader = trainloader
    time_step = 10
    reg_param = 0.5
    for epoch in range(num_epochs):
        for imgs, target in autoencoder_train_dataloader:
            #### For N-Caltech/N-Cars #####
            model_input_size = torch.tensor([256, 256])
            imgs=imgs.squeeze(2)
            imgs = torch.nn.functional.interpolate(imgs, torch.Size(model_input_size))
            imgs=imgs.unsqueeze(2)
            #imgs=imgs.repeat(1,1,3,1,1)
            #############################
            if len(imgs)<64:
              continue
            l1 = 0
            running_loss = 0
            correct = 0
            total=0
            acc=0
            a=0
            for step in range(time_step):
                x = imgs[:, step]
                x /= x.max()
                target = target.long()
                x, target = x.to(device), target.to(device)
                output= autoencoder(x)            
                loss = distance(output, x)
                l1 += 1
                optimizer.zero_grad()
                reg_alpha = torch.tensor(1.).cuda()
                a_lambda = torch.tensor(0.0001).cuda()
                model_children = list(autoencoder.children())
                alpha = []
                for name, param in autoencoder.named_parameters():
                    if 'alpha' in name:
                      alpha.append(param.item())
                      reg_alpha += param.item() ** 2
                loss += a_lambda * (reg_alpha)
                ###l1_loss = sparse_loss(autoencoder, x, model_children).cuda()
                ## add the sparsity penalty
                ###loss = loss + reg_param * l1_loss
                ####

                ####### Group Lasso
                thre1 = autoencoder.get_global_mp_thre(ratio=0.90)
                lamda = torch.tensor(2.5).cuda()
                reg_g1 = torch.tensor(0.).cuda()
                reg_linear = torch.tensor(0.).cuda()
                group_ch = 4
                thre_list = []
                penalty_groups = 0
                count = 0
                lin_count = 0
                for m in autoencoder.modules():
                    if isinstance(m, QConv2d):
                        if count in [1]:
                            w_l = m.weight
                            kw = m.weight.size(2)
                            if kw != 1:
                                num_group = w_l.size(0) * w_l.size(1) // group_ch
                                w_l = w_l.view(w_l.size(0), w_l.size(1) // group_ch, group_ch, kw, kw)
                                w_l = w_l.contiguous().view((num_group, group_ch, kw, kw))
                #            
                                # reg1, thre1, penalty_group1 = glasso_thre(w_l, 1, args.ratio)
                                reg1, penalty_group1 = glasso_global_mp(w_l, dim=1, thre=thre1)
                                reg_g1 += reg1
                                thre = thre1
                                penalty_group = penalty_group1
                        
                        count += 1
                loss += lamda * (reg_g1)  
                running_loss += loss.item() 
                #################################    
                loss.backward()
                optimizer.step()
            loss1 = running_loss / l1
            image=x
            #import pdb;pdb.set_trace()
        print('epoch {:d}/{:d}, loss: {:.4f}'.format(epoch + 1, num_epochs, loss1), end='\r')
    autoencoder.encoder.register_forward_hook(get_activation('encoder'))
    output = autoencoder(x)
    layer_out=activation['encoder']
    #import pdb;pdb.set_trace()
    #print(layer_out.shape)
    #print(layerout.keys())
    #layer1=layerout['relu2']
    print("===========")
    count_non=len(torch.nonzero(layer_out.view(-1)))
    print(count_non)
    #print("===========")
    count_t=len(layer_out.view(-1))
    print(count_t)
    out_sparsity = 1-count_non/count_t
    print(out_sparsity)
    print("===========")

    count=0
    all_num = 0.0
    count_num_one = 0.0
    for m in autoencoder.modules():
        if isinstance(m, QConv2d):
            if not count in [0] and not m.weight.size(2) == 1:
                w_mean = m.weight.mean()
                w_l = m.weight
                w_l, _, _ = odd_symm_quant(w_l, nbit=4, mode='mean', k=2)

                kw = m.weight.size(2)

                count_num_layer = w_l.size(0) * w_l.size(1) * kw * kw
               
                all_num += count_num_layer
                count_one_layer = len(torch.nonzero(w_l.view(-1)))
                count_num_one += count_one_layer
                #import pdb;pdb.set_trace()
                #num_group = w_l.size(0) * w_l.size(1) // group_ch
                #w_l = w_l.view(w_l.size(0), w_l.size(1) // group_ch, group_ch, kw, kw)
                #w_l = w_l.contiguous().view((num_group, group_ch * kw * kw))

                #grp_values = w_l.norm(p=2, dim=1)
                #non_zero_idx = torch.nonzero(grp_values)
                #num_nonzeros = len(non_zero_idx)

                #all_group += num_group
                #all_nonsparse += num_nonzeros

            count += 1
    overall_sparsity = 1 - count_num_one / all_num
    #group_sparsity, overall_sparsity, sparse_group = get_weight_sparsity(autoencoder)
    del distance
    del optimizer
#def Test(testdata, model1, model2):
    time_step = 1
    correct = 0
    total = 0
    acc = 0
    a = 0
    autoencoder = autoencoder.to('cpu')
    #autoencoder_test_dataloader = testdata
    autoencoder_test_dataloader=testdata
    variation=nn.MSELoss()
    Image_diff=torch.Tensor([])
    #classifier = model1.to(device)
    #autoencoder = model2.to(device)
    for imgs, labels in autoencoder_test_dataloader:
        ####### For N-Caltech/ N-Cars Dataset ##########
        model_input_size = torch.tensor([256, 256])
        imgs=imgs.squeeze(2)
        imgs=torch.nn.functional.interpolate(imgs, torch.Size(model_input_size))
        imgs=imgs.unsqueeze(2)
        #imgs=imgs.repeat(1,1,3,1,1)
        #####################################
        #target = int(target/target.max())
        for step in range(time_step):
            x = imgs[:, step]
            x /= x.max()
            #x, target = x.to(device), labels.to(device)
            ae_imgs = autoencoder(x)
            ######## For N-Caltech dataset
            #import pdb;pdb.set_trace()
            ######################################
            ae_imgs = img_quant(ae_imgs)
            diff=variation(x,ae_imgs)
            diff=torch.Tensor([diff])
            print(diff)
            #model_input = torch.tensor([3, 64, 64])
            #imgs=ae_imgs.squeeze(2)
            #imgs = torch.nn.functional.interpolate(imgs, torch.Size(model_input))
            #ae_imgs=imgs.unsqueeze(2)
            output = classifier(ae_imgs)
            #import pdb;pdb.set_trace()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #img1 = ae_imgs[0]
            #nozero1 = img1[img1 > 0]
            #print(len(nozero1))
            #print("==============")
            #save_imgs('Image of 4-bit precision_{}_{:d}.png'.format(4, 128), x, ae_imgs)
            save_imgs('N_MNIST Encoder_Decoder', x, ae_imgs)
        Image_diff= torch.cat((Image_diff,diff), 0)
        a += 1
        acc += 100 * correct / total
    Image_difference = Image_diff.mean()
    # save some images
    # save_imgs('ae_samples_{}_{:d}.png'.format(quantize_bits, encoding_dims), imgs, ae_imgs)
    accuracy = acc / a
    results.append(accuracy)
    return accuracy, overall_sparsity,Image_difference

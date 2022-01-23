"""
MNIST-DVS dataset
Data loader

http://www2.imse-cnm.csic.es/caviar/MNISTDVS.html
"""
import torch
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import torch
from torch.utils.data import Dataset

def get_mnist_dvs(root, batch_size):
    ### For N-Cars#####
    #train_dir = root + "train_cars.pkl"
    #valid_dir = root + "valid_cars.pkl"
    #test_dir = root + "test_cars.pkl"
    ###################
    ### For N-MNIST####
    train_dir = root + "train.pkl"
    valid_dir = root + "valid.pkl"
    test_dir = root + "test.pkl"
    ###################

    with open(train_dir, 'rb') as f:
        train_set = pickle.load(f)
    
    with open(valid_dir, 'rb') as f:
        valid_set = pickle.load(f)

    with open(test_dir, 'rb') as f:
        test_set = pickle.load(f)
    
    image_transforms=transforms.Compose([
    transforms.Grayscale(num_output_channels=3)
    ])
    
    ############## Custom data making ############
    #class CustomTensorDataset(Dataset):
    #  def __init__(self, dataset, label, transform_list=None):
    #    data_X = dataset
    #    data_y = label
    #    X_tensor, y_tensor = torch.tensor(data_X), torch.tensor(data_y)
    #    #X_tensor, y_tensor = Tensor(data_X), Tensor(data_y)
    #    tensors = (X_tensor, y_tensor)
    #    assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
    #    self.tensors = tensors
    #    self.transforms = transform_list

    #  def __getitem__(self, index):
    #    x = self.tensors[0][index]

    #    if self.transforms:
    #      #for transform in self.transforms: 
    #      #  x = transform(x)
    #      x = self.transforms(x)

    #    y = self.tensors[1][index]

    #    return x, y

    #  def __len__(self):
    #    return self.tensors[0].size(0)
        
    #trainset = train_set[0]
    #trainlabel = train_set[1]
    #validset = valid_set[0]
    #validlabel = valid_set[1]
    #testset = test_set[0]
    #testlabel = test_set[1]
    #transforms_list = transforms.Compose([
    #          #transforms.RandomCrop(im_size, padding=4),
    #          #transforms.RandomHorizontalFlip(),
    #          #transforms.ToPILImage(),
    #          transforms.Grayscale(num_output_channels=1),
    #          #transforms.Resize((im_size, im_size)),
    #          transforms.ToTensor(),
    #          ])
    #if transforms_list: # != None
    #  train_dataset = CustomTensorDataset(dataset=trainset, label= trainlabel, transform_list=transforms_list)
    #  val_dataset = CustomTensorDataset(dataset=validset, label= validlabel, transform_list=transforms_list)
    #  test_dataset = CustomTensorDataset(dataset=testset,  label= testlabel, transform_list=transforms_list)
    #else:
    #  train_dataset = CustomTensorDataset(dataset=trainset, label= trainlabel)
    #  val_dataset = CustomTensorDataset(dataset=validset, label= validlabel)
    #  test_dataset = CustomTensorDataset(dataset=testset,  label= testlabel)
    #####################################################
    # wrap up the dataset
    #### For N-Cars and N-Caltech ##########
    t_dataset = TensorDataset(train_set[0], train_set[1])
    #import pdb;pdb.set_trace()
    ##data=torch.utils.data.random_split(t_dataset, [7547, 1000], generator=torch.Generator().manual_seed(42))
    data=torch.utils.data.random_split(t_dataset, [7000, 1000], generator=torch.Generator().manual_seed(42))
    train_dataset=data[0]
    test_dataset=data[1]
    ########################################
    
    ##### For N-Caltech and N-Cars #########
    ###train_dataset = TensorDataset(train_set[0], train_set[1])
    #val_dataset = torch.split(train_dataset,)
    #train_bb = train_set[2]
    ##val_dataset = TensorDataset(valid_set[0], valid_set[1])
    #valid_bb = valid_set[2]
    ###test_dataset = TensorDataset(test_set[0], test_set[1])
    #test_bb = test_set[2]
    ########################################
    
    # create data loader
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    ##valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    #return train_loader, valid_loader, test_loader, train_bb, valid_bb, test_bb
    #return train_set, valid_loader, test_set, train_bb, valid_bb, test_bb
    ## Original MNIST-DVS
    return train_loader, test_loader
"""
Read the DVS raw measurement then convert to Pytorch Tensor

Jian Meng

Arizona State University
"""

import os
import numpy as np
import torch
import torchvision.transforms.functional as F
import pickle
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
from read_events import load_dvs_events
from read_events import load_atis_events
from sklearn import preprocessing
#from events_tfds.data_io.neuro import load_neuro_events
from neuro import load_neuro_events

RED = np.array(((255, 0, 0)), dtype=np.uint8)
GREEN = np.array(((0, 255, 0)), dtype=np.uint8)
WHITE = np.array(((255, 255, 255)), dtype=np.uint8)
BLACK = np.array(((0, 0, 0)), dtype=np.uint8)
GREY = np.array(((220, 220, 220)), dtype=np.uint8)

def get_frames(
    coords,
    time,
    polarity=None,
    dt=None,
    num_frames=None,
    shape=None,
    flip_up_down=False
):
    r"""
    convert the events to rgb frames
    """
    assert time.size > 0, "the length of the time sequence must greater than 0!"
    t_start = time[0]
    t_end = time[-1]
    
    if dt is None:
        dt = int((t_end - t_start) // (num_frames - 1))
    else:
        num_frames = (t_end - t_start) // dt + 1
        
    if shape is None:
        shape = np.max(coords, axis=0)[-1::-1] + 1    # [-1::-1] quickly reverse
        max_val = np.max(coords, axis=0)
    else:
        shape = shape[-1::-1]

    frame_data = np.zeros((num_frames, *shape, 3), dtype=np.uint8)

    if polarity is None:
        color = GREY
    else:
        colors = np.where(polarity[:, np.newaxis], RED, GREEN)
    i = np.minimum((time-t_start) // dt, num_frames - 1)
    x, y = coords.T
    
    frame_data[(i, y, x)] = colors
    if flip_up_down:
        y = shape[0] - y - 1
    #plt.imshow(frame_data[2])
    #plt.savefig('test.png')
    #import pdb;pdb.set_trace()
    return frame_data

def frame2grayscale(frames):
    r"""
    convert the rgb frames to grayscale tensor
    """
    tensor = torch.from_numpy(frames)
    tensor = tensor.permute(0,3,1,2)
    #tensor = F.rgb_to_grayscale(tensor) # convert the rgb frames to gray scale

    #assert tensor.size(1) == 1, "number of channels must be 1!"

    return tensor

def get_event(folder_name, nframes, label, subfolder):
#def get_event(archive):
    r"""
    Get the data for a single digit (class)

    Convert each single event to tensors with dim = (nframes, 128, 128, 1)
    """
    dataset = torch.Tensor([])       
    #lab = [folder_name]
    #le = preprocessing.LabelEncoder()
    #targets = le.fit_transform(lab)
    #label = torch.as_tensor(targets)
    label = label
    #folder = os.path.join(folder_name, "scale{}".format(scale))
    folder = folder_name
    #path=path
    bounding_box_list = torch.Tensor([])
    fold_name=subfolder
    # loop over the data
    for idx, filename in enumerate(os.listdir(folder)):
        #annotation_file = 'annotation' + filename[5:]
        example_id = int(filename[4:-7])
        #print('count')
        with open(os.path.join(folder, filename), "rb") as fp:
            #time, coords, polarity = load_neuro_events(fp)
            time, coords, polarity = load_atis_events(fp)
            #time, x, y, polarity = load_dvs_events(fp)
            #coords = np.stack((x, y), axis=-1)
            
            # generate the frames based on the current event
            frames = get_frames(coords=coords, time=time, polarity=polarity, num_frames=nframes)
        
            # convert the numpy frames to grayscale tensors
            image = frame2grayscale(frames)
            #import pdb;pdb.set_trace()
            
            image = image.unsqueeze(0)
            x, y = image.size(3), image.size(4)
            dif_x = 100 - x
            dif_y = 120 - y
            if dif_x > 0:
                    image = torch.nn.functional.pad(image, (0,0,0,abs(dif_x),0,0,0,0,0,0))
            elif dif_x < 0:
                    image = torch.nn.functional.pad(image, (0,0,0,abs(dif_x),0,0,0,0,0,0))
                    
            if dif_y > 0:
                    image = torch.nn.functional.pad(image, (0,abs(dif_y),0,0,0,0,0,0,0,0)) 
            elif dif_y < 0:
                    image = torch.nn.functional.pad(image, (0,abs(dif_y),0,0,0,0,0,0,0,0))
            # pad the imperfect measurements
            if len(dataset.size()) > 1:
                x, y = image.size(3), image.size(4)
                diff_x = dataset.size(3) - x
                diff_y = dataset.size(4) - y

                if diff_x > 0:
                    image = torch.nn.functional.pad(image, (0,0,0,abs(diff_x),0,0,0,0,0,0))
                elif diff_x < 0:
                    image = torch.nn.functional.pad(image, (0,0,0,abs(diff_x),0,0,0,0,0,0))
                    
                if diff_y > 0:
                    image = torch.nn.functional.pad(image, (0,abs(diff_y),0,0,0,0,0,0,0,0)) 
                elif diff_y < 0:
                    image = torch.nn.functional.pad(image, (0,abs(diff_y),0,0,0,0,0,0,0,0)) 
                #import pdb;pdb.set_trace()     
            try:
                dataset = torch.cat((dataset, image), dim=0)
            except:
                import pdb;pdb.set_trace()
        #with open(os.path.join(path, fold_name, annotation_file), "rb") as fa:
        #    annotations = np.fromfile(fa.read(), dtype=np.uint8)
        #    annotations=annotations[2:10]
        ##bb=readBoundingBox(os.path.join(path, fold_name, annotation_file))
        ##bb=torch.Tensor([bb])
        #bounding_box_list=bounding_box_list.append(bb)
        ##bounding_box_list = torch.cat((bounding_box_list,bb),0)
    #import pdb;pdb.set_trace()    
    # normalize the data 
    labels = torch.ones(dataset.size(0)).mul(label)
    #bounding_box_list=torch.Tensor(bounding_box_list)
    train_set, valid_set, test_set = torch.split(dataset, (len(dataset)-30, 13, 17), dim=0)
    #import pdb;pdb.set_trace()
    train_label, valid_label, test_label = torch.split(labels, (len(labels)-30, 13, 17), dim=0) 
    #bb_train_list, bb_valid_list, bb_test_list = torch.split(bounding_box_list, (len(bounding_box_list)-30, 13, 17), dim=0)

    return train_set, train_label, valid_set, valid_label, test_set, test_label

#def readBoundingBox(file_path):
#        f = open(file_path)
#        annotations = np.fromfile(f, dtype=np.int16)
#        f.close()
#        return annotations[2:10]
        
def get_dataset(root, nframes):
    train_data, train_labels, train_bb = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    valid_data, valid_labels, valid_bb = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    test_data, test_labels, test_bb = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

    #for ii in tqdm(range(10)):
    #    folder_name = os.path.join(root, "grabbed_data{}".format(ii))
    #    train_set, train_label, valid_set, valid_label, test_set, test_label = get_event(folder_name=folder_name, nframes=nframes)

        # cascading
    #    train_data = torch.cat((train_data, train_set), dim=0)
    #    train_labels = torch.cat((train_labels, train_label), dim=0)

    #    valid_data = torch.cat((valid_data, valid_set), dim=0)
    #    valid_labels = torch.cat((valid_labels, valid_label), dim=0)

    #    test_data = torch.cat((test_data, test_set), dim=0)
    #    test_labels = torch.cat((test_labels, test_label), dim=0)
    #path1 = os.path.join(root,'Caltech101')
    path1 = root
    #path2 = os.path.join(root,'Caltech101_annotations')
    directory_contents = os.listdir(path1)
    print(directory_contents)
    l=0
    for item in directory_contents:
        #print(item)
        l = l+1
        folder_name = os.path.join(path1, item)
        print("One File Done")
        #folder_name = item
        train_set, train_label, valid_set, valid_label, test_set, test_label = get_event(folder_name=folder_name, nframes=nframes, label=l,subfolder=item)

        # cascading
        train_data = torch.cat((train_data, train_set), dim=0)
        train_labels = torch.cat((train_labels, train_label), dim=0)
        #train_bb = torch.cat((train_bb, bb_train_list), dim=0)
        
        valid_data = torch.cat((valid_data, valid_set), dim=0)
        valid_labels = torch.cat((valid_labels, valid_label), dim=0)
        #valid_bb = torch.cat((valid_bb, bb_valid_list), dim=0)

        test_data = torch.cat((test_data, test_set), dim=0)
        test_labels = torch.cat((test_labels, test_label), dim=0)
        #test_bb = torch.cat((test_bb, bb_test_list), dim=0)

    print("Size of training samples = {}\n".format(list(train_data.size())))
    print("Size of valid samples: {}\n".format(list(valid_data.size())))
    print("Size of test samples: {}\n".format(list(test_data.size())))

    # save the file to .pkl
    print("Saving training set...")
    with open("train_cars.pkl", 'wb') as f:
        pickle.dump((train_data, train_labels), f, protocol=4)
    f.close()

    print("Saving valid set...")
    with open("valid_cars.pkl", 'wb') as f:
        pickle.dump((valid_data, valid_labels), f, protocol=4)
    f.close()
    
    print("Saving test set...")
    with open("test_cars.pkl", 'wb') as f:
        pickle.dump((test_data, test_labels), f, protocol=4)
    f.close()
    


if __name__ == "__main__":
    DIR = "n-cars_test"
    #DIR = "/home/ahasssan/ahmed/rpg_asynet/data/"
    get_dataset(DIR, nframes=20)

    # load the dataset to verify
    with open("train_cars.pkl", 'rb') as f:
    #with open("train_caltech.pkl", 'rb') as f:
        train_set = pickle.load(f)

    train_set = tuple([train_set[0], train_set[1].long()])
    print("Dataset loaded! size = {}; type = {}".format(train_set[0].size(), type(train_set[0])))
    print("Unique labels = {}".format(torch.unique(train_set[1])))
    print("No of Unique labels = {}".format(len(torch.unique(train_set[1]))))
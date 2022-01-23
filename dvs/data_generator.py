   
"""
Load the N-CARs dataset to event
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import DatasetFolder
from utils_.visualizations import visualizeHistogram
from torch.utils.data import random_split, DataLoader

# RGB characteristics
RED = np.array(((255, 0, 0)), dtype=np.uint8)
GREEN = np.array(((0, 255, 0)), dtype=np.uint8)
WHITE = np.array(((255, 255, 255)), dtype=np.uint8)
BLACK = np.array(((0, 0, 0)), dtype=np.uint8)
GREY = np.array(((220, 220, 220)), dtype=np.uint8)


# DVS event characteristics
EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event

# configuration for N-Cars event
x_mask = 0x00003FFF
y_mask = 0x0FFFC000
pol_mask = 0x10000000
x_shift = 0
y_shift = 14
pol_shift = 28

polarity_mask = 1
polarity_shift = None

valid_mask = 0x80000000
valid_shift = 31

def read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr

def frame2grayscale(frames):
    r"""
    convert the rgb frames to grayscale tensor
    """
    tensor = torch.from_numpy(frames)
    tensor = tensor.permute(0,3,1,2)
    return tensor

def load_neuro_events(fobj):
    """
    Load events from file.
    File stores concatenated events. Each occupies 40 bits as described below:
        bit 39 - 32: Xaddress (in pixels)
        bit 31 - 24: Yaddress (in pixels)
        bit 23: Polarity (0 for OFF, 1 for ON)
        bit 22 - 0: Timestamp (in microseconds)
    Args:
        fobj: file-like object with a `read` method.
    Returns:
        Event stream, namedtuple with names/shapes:
            time: [num_events] int64
            coords: [num_events, 2] uint8
            polarity: [num_events] bool
    """
    raw_data = np.fromstring(fobj.read(), dtype=np.uint8)
    raw_data = raw_data.astype(np.uint32)
    x = raw_data[::5]
    y = raw_data[1::5]
    polarity = ((raw_data[2::5] & 128) >> 7).astype(np.bool)
    time = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    valid = y != 240

    x = x[valid]
    y = y[valid]
    polarity = polarity[valid]
    time = time[valid].astype(np.int64)

    return x, y, time, polarity

def load_atis_events(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "%":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    fp.seek(p + 2)
    data = np.fromstring(fp.read(), dtype="<u4")

    time = data[::2]
    coords = data[1::2]

    x = read_bits(coords, x_mask, x_shift)
    y = read_bits(coords, y_mask, y_shift)
    pol = read_bits(coords, pol_mask, pol_shift)
    return x, y, time, pol


class DVSLoader:
    def __init__(self, root:str, mode:str, height:int, width:int, save_dir:str):
        r"""
        Dataloader for NCARS dataset
        """
        self.root = root
        self.mode = mode
        self.height = height
        self.width = width
        
        # assert self.mode in ["train", "test"], "the target folder must be either training or test"

        # gloabal directory
        self.root = os.path.join(self.root, self.mode)
        self.folders = os.listdir(self.root)

        # save path
        self.save_root = os.path.join(save_dir, self.mode)
        if not os.path.isdir(self.save_root):
            os.makedirs(self.save_root)

    def generate_event_histogram(self, events, shape):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        H, W = shape
        x, y, t, p = events.T
        x = x.astype(np.int)
        y = y.astype(np.int)

        img_pos = np.zeros((H * W,), dtype="float32")
        img_neg = np.zeros((H * W,), dtype="float32")

        np.add.at(img_pos, x[p == 1] + W * y[p == 1], 1)
        np.add.at(img_neg, x[p == -1] + W * y[p == -1], 1)

        histogram = np.stack([img_neg, img_pos], -1).reshape((H, W, 2))

        return histogram

    def get_frames(
        self,
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

        # import pdb;pdb.set_trace()
        
        frame_data[(i, y, x)] = colors
        if flip_up_down:
            y = shape[0] - y - 1
        return frame_data


    def event2hist(self, save=False):
        r"""
        Read the events from the folder
        """
        for item in self.folders:
            class_folder = os.path.join(self.root, item)
            for ii, filename in enumerate(os.listdir(class_folder)):
                with open(os.path.join(class_folder, filename), "rb") as fp:
                    x, y, time, pol = load_atis_events(fp)
                    event = np.stack([x, y, time, pol], axis=-1).astype('float64')

                    # visualize event
                    hist = self.generate_event_histogram(event, shape=(self.height, self.width))   # convert event to hist
                    vhist = visualizeHistogram(histogram=hist, path_name=None)
                    vhist = torch.from_numpy(vhist)
                    vhist = vhist.permute(2,0,1)
                    print(f"class={item}, idx={ii}; shape={list(vhist.size())}")

                    if save:
                        save_dir = os.path.join(self.save_root, item)
                        if not os.path.isdir(save_dir):
                            os.makedirs(save_dir)
                        
                        torch.save(vhist, save_dir+"/%04d.pt"%ii)   
            
    def event2quene(self, nframes, save=False, get_hist=False):
        r"""
        Load sychronous event
        """
        l = 0
        labels = []
        cars = 0
        for item in self.folders:
            class_folder = os.path.join(self.root, item)
            for ii, filename in enumerate(os.listdir(class_folder)):
                with open(os.path.join(class_folder, filename), "rb") as fp:
                    x, y, time, pol = load_atis_events(fp)
                    # x, y, time, pol = load_neuro_events(fp)
                    coords = np.stack((x, y), axis=-1)
                    frames = self.get_frames(coords=coords, time=time, polarity=pol, num_frames=nframes)
                    
                    # visualize event
                    if get_hist and cars < 20:
                        if not "background" in class_folder:
                            f = frames / 255.
                            f = np.sum(f, axis=0)
                            f = f / f.max()
                            cars += 1
                            plt.imshow(f)
                            plt.savefig(f'./hist_figs/quene_{ii}.png')

                    # tensors
                    frames = torch.from_numpy(frames)
                    diff_x = self.height - frames.size(1)
                    diff_y = self.width - frames.size(2)
                    
                    if diff_x > 0:
                        frames = torch.nn.functional.pad(frames, (0,0,0,0,0,diff_x,0,0))
                    
                    if diff_y > 0:
                        frames = torch.nn.functional.pad(frames, (0,0,0,diff_y,0,0,0,0))
                    
                    frames = frames.permute(0,3,1,2)
                    print(f"class={item}, idx={ii}; shape={list(frames.size())}")

                    if save:
                        save_dir = os.path.join(self.save_root, item)
                        if not os.path.isdir(save_dir):
                            os.makedirs(save_dir)
                        
                        torch.save(frames, save_dir+"/%04d.pt"%ii)

def loadpt(data):
    return torch.load(data)

def get_caltect_loader(path, batch_size):
    dataset = DatasetFolder(
        root=path,
        loader=loadpt,
        extensions=(".pt")
    )

    num_classes=102

    datasize = len(dataset)
    trainsize = int(0.8*datasize)
    testsize = datasize - trainsize
    trainset, testset = random_split(dataset, [trainsize, testsize], generator=torch.Generator().manual_seed(42))
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader, num_classes

def get_ncars_loader(path, batch_size):
    num_classes = 2
    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")
    
    trainset = DatasetFolder(root=train_path, loader=loadpt, extensions=(".pt"))
    testset = DatasetFolder(root=test_path, loader=loadpt, extensions=(".pt"))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader, num_classes



if __name__ == "__main__":
     #root = "/home/ahasssan/ahmed/rpg_asynet/data/n-cars/"
     #save_dir = "/home/ahasssan/ahmed/rpg_asynet/data/ncars_pt/"
     #mode = "train"
     #height = 100
     #width = 120

     #loader = DVSLoader(root, mode, height=height, width=width, save_dir=save_dir)
#     loader.event2quene(nframes=30, save=True)
     #loader.event2hist(save=True)

     # pytorch dataset
     data_set = DatasetFolder(
         root="/home/ahasssan/ahmed/rpg_asynet/data/ncars_pt/test/",
         loader=loadpt,
         extensions=("pt")
     )

     c = data_set.find_classes("/home/ahasssan/ahmed/rpg_asynet/data/ncars_pt/test/")
     print(c[0])
     sample, target = data_set.__getitem__(0)
     print(sample.unique())
     print(data_set.__len__)

    # root = "/home/jmeng15/Caltech101/"
    # save_dir = "/home/jmeng15/data/Caltech101_pt/"
    # mode=""
    # height, width = 180, 240

    # loader = DVSLoader(root, mode, height=height, width=width, save_dir=save_dir)
    # loader.event2quene(nframes=30, save=True)

    # # pytorch dataset
    # data_set = DatasetFolder(
    #     root="/home/jmeng15/data/Caltech101_pt/",
    #     loader=loadpt,
    #     extensions=(".pt")
    # )

#     datasize = len(data_set)
#     trainsize = int(0.8*datasize)
#     testsize = datasize - trainsize
#     trainset, testset = random_split(data_set, [trainsize, testsize], generator=torch.Generator().manual_seed(42))
    
#     print(len(trainset))
#     print(len(testset))

#     # sample, target = data_set.__getitem__(0)
#     # print(sample.unique())
#     # print(data_set.__len__)
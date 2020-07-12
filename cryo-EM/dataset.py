import torch
import random
import linecache
import numpy as np
from torch.utils.data import Dataset
from PIL import Image



class SiameseDataset(Dataset):

    def __init__(self, txt, labellist, transform=None):
        self.transform = transform
        self.txt = txt
        self.labellist = labellist
  
    def __getitem__(self, index):
        idx0 = random.randint(0,self.__len__()-1)
        label0 = self.labellist[idx0]
        line0 = linecache.getline(self.txt, idx0+1).strip('\n')
        img0 = Image.open(line0)

        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                idx1 = random.randint(1,self.__len__())
                if self.labellist[idx1-1] == label0:
                    break
        else:
            idx1 = random.randint(1,self.__len__())
        label1 = self.labellist[idx1-1]
        line1 = linecache.getline(self.txt, idx1).strip('\n')
        img1 = Image.open(line1)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1, torch.from_numpy(np.array([int(label0!=label1)],dtype=np.float32))

    def __len__(self):
        with open(self.txt, 'r') as f:
            num = len(f.readlines())
        return num


class Eval_Dataset(Dataset):

    def __init__(self, txt, transform=None, initial=False):
        self.transform = transform
        self.txt = txt
        self.initial = initial

    def __getitem__(self, index):
        line = linecache.getline(self.txt, index+1).strip('\n')
        img = Image.open(line)

        if not self.initial:
            img = img.convert("L")

        if self.transform is not None:
            img = self.transform(img)
        
        return img

    def __len__(self):
        with open(self.txt, 'r') as f:
            num = len(f.readlines())
        return num  
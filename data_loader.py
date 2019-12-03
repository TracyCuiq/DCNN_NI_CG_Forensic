import torch
from torch.utils.data import  Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import os
import torchvision
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


class CGPIM_Data(Dataset):

    def __init__(self, root, transform=None, target_transform=None, final_img_size=224, isTrain=True):
        
        self.root = root
        imgs = []
        with open(root) as f:
            for line in f:
                line = line.strip('\n\r').strip('\n').strip('\r')
                words = line.split(' ')
                imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.final_img_size = final_img_size     
        self.transform2 = transforms.Compose([transforms.Resize((final_img_size, final_img_size)), transforms.ToTensor(), ])
        self.isTrain = isTrain
        self.train_root = ''
        self.test_root = ''
        self.valid_root = ''

    def __getitem__(self, index):

        fn, label = self.imgs[index]
        _root_dir = self.train_root if self.isTrain else self.test_root
        img = Image.open(os.path.join(_root_dir, fn)).convert('L')                
        if self.transform is not None:
            img = self.transform(img)
        return(img, int(label))

    def __len__(self):
        return len(self.imgs)

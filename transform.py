import random
import torch
from torchvision import transforms
from PIL import Image,ImageOps,ImageFilter

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self,img):
        ratio = self.size[0]/self.size[1]
        w,h = img.size
        if w/h < ratio:
            t = int(h*ratio)
            w_padding = (t-w)//2
            img = img.crop((-w_padding,0,w+w_padding,h))
        else:
            t = int(w*ratio)
            h_padding = (t-h)//2
            img = img.crop((0,-h_padding,w,h+h_padding))
        img = img.resize(self.size,self.interpolation)
        return img



def get_train_transform(size):
    return transforms.Compose([
        Resize((int(size*(256/224)),int(size*(256/224)))),
        transforms.RandomCrop(size),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    

def get_test_transform(size):
    return transforms.Compose([
        Resize((int(size*(256/224)),int(size*(256/224)))),
        transforms.CenterCrop(size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])


def get_transforms(input_size=224, test_size=224, backbone=None):
    
    transformations = {}
    transformations['val_train'] = get_train_transform(input_size)
    transformations['val_test'] = get_test_transform(test_size)
    transformations['test'] = get_test_transform(test_size)
    return transformations
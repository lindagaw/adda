from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

def obtain_office_31(subdomain):
    """Get MNIST dataset loader."""
    # image pre-processing
    batch_size = 16
    image_size = 299

    dataroot_amazon = "..//dcgan//datasets//office-31-intact//amazon//images//"
    dataroot_dslr = "..//dcgan//datasets//office-31-intact//dslr//images//"
    dataroot_webcam = "..//dcgan//datasets//office-31-intact//webcam//images//"

    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #AddGaussianNoise(0., 1.)
        ])

    if subdomain == 'A':
        dataroot = dataroot_amazon
    elif subdomain == 'D':
        dataroot = dataroot_dslr
    elif subdomain == 'W':
        dataroot = dataroot_webcam

    dataset = datasets.ImageFolder(root=dataroot, transform=transform)
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

    dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


    return dataloader_train, dataloader_test

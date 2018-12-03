import torch
import os
import matplotlib.pyplot as plt
import pickle
import argparse
from utils import test as Test, load_dictionary_from_pickle
from models.resnet import ResNet18
from models.vgg import VGG
from models.lenet import *
from models.alexnet import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# test 1: Print epoch and accuracy of the checkpoint
resume = './checkpoint/resnet_imagenet/ckpt_trial_1_epoch_300.t7'
print("==> loading checkpoint '{}'".format(resume))
if device == 'cpu':
    checkpoint = torch.load(resume, map_location='cpu')
else:
    checkpoint = torch.load(resume)
print("checkpoint keys:{}".format(checkpoint.keys()))
start_epoch = checkpoint['epoch']
accuracy = checkpoint['acc']
print("==>epoch {}, accuracy {}%".format(start_epoch,accuracy))

# test 2: Load data from pickle
net = AlexNet()
filename = 'alexnet_cifar.pkl'
criterion = F.nll_loss
#criterion = nn.CrossEntropyLoss()
print("==>let's test!")
deepfoolLoader, outputs = load_dictionary_from_pickle(filename)
accuracy2, activs2, targets2 = Test(net, deepfoolLoader, 'cpu', criterion, 10)  # Todo : how to apply cuda?

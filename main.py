from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import os
import argparse
from models.lenet import *
from models.alexnet import *
from models.resnet import *
from models.vgg import *
from utils import progress_bar
import numpy as np
import h5py
from utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
parser.add_argument('--resume_epoch', default=20, type=int, help='resume from epoch')
parser.add_argument('--save_every', default=10, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 1 or last checkpoint epoch
image_channel = 0   # channel of the image (1 or 3)
# height/width of the image (in mnist, it is 28*28 while in imagenet and cifar, it is 32*32)
image_height = 0    
image_width = 0

if args.dataset=='cifar':
    trainloader,  testloader = prepare_cifar10()
    criterion = nn.CrossEntropyLoss()
    image_height = 32
    image_width = 32
    image_channel = 3
elif args.dataset=='imagenet':
    trainloader,  testloader = prepare_tiny_imagenet()
    criterion = nn.CrossEntropyLoss()
    image_height = 32
    image_width = 32
    image_channel = 3
elif args.dataset=='mnist':
    trainloader,  testloader = prepare_mnist()
    criterion = F.nll_loss
    image_height = 28
    image_width = 28
    image_channel = 1
    
oname = args.net + '_' + args.dataset

# Model
print('==> Building model..')
if args.net=='lenet' and args.dataset=='mnist':
    num_classes = 10
    net = LeNet(num_classes=10)
elif args.net=='vgg' and args.dataset=='cifar':
    num_classes = 10
    net = VGG('VGG16')
elif args.net=='vgg' and args.dataset=='imagenet':
    num_classes = 200
    net = VGG('VGG16', num_classes=200)
elif args.net=='resnet' and args.dataset=='cifar':
    num_classes = 10
    net = ResNet18()
elif args.net=='resnet' and args.dataset=='imagenet':
    num_classes = 200
    net = ResNet18(num_classes=200)
elif args.net=='alexnet' and args.dataset=='cifar':
    num_classes = 10
    net = AlexNet(num_classes=10)
elif args.net=='alexnet' and args.dataset=='imagenet':
    num_classes = 200
    net = AlexNet(num_classes=200)
    
net = net.to(device)

print(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
        
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if device == 'cuda':
        checkpoint = torch.load(
            './checkpoint/' + args.net + '_' + args.dataset + '/ckpt_trial_' + str(args.trial) + '_epoch_' + str(
                args.resume_epoch) + '.t7')
    else:
        checkpoint = torch.load(
            './checkpoint/' + args.net + '_' + args.dataset + '/ckpt_trial_' + str(args.trial) + '_epoch_' + str(
                args.resume_epoch) + '.t7'
            , map_location="cpu")  # notice: map_location is changed with no-cuda machine

    state_dict = checkpoint['net']
    if device == 'cpu':
        # remove modules. from dataParallel function (it can work since num_worker=2)
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict, strict=True) # with no data parallel
    else:
        net.load_state_dict(state_dict, strict=True)

    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
        
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, mode='max', verbose=True)

''' Create dataset for dumping activations '''
# file = h5py.File('./activations/'+oname+'/activations_trial_' + str(args.trial) + '.hdf5', 'a')

accuracy, activs, targets = test(net, testloader, device, criterion, n_test_batches=10)

deepfoolLoader,outputs = deep_fool_images(net, testloader, num_classes)
# deepfoolLoader,outputs = deep_fool_images(net, testloader_array[0], device)
accuracy2, activs2, targets2 = test(net, deepfoolLoader, device, criterion, n_test_batches=10)

# acc = deep_fool_test(net,outputs,criterion) # lightweight method to analyze accuracy
########### show images  ###########
# print("==> Show images...")
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(32, 32))
# [row, col] = (5, 4)
# print(outputs.__len__())
# for batch_idx, (inputs, targets) in enumerate(testloader_array[0]):
#     if(batch_idx>=row*col/2):
#         break
#     fig.add_subplot(row, col, 2 * batch_idx + 1)
#     plt.imshow(inputs[batch_idx][0],cmap ='gray')
#     # plt.imsave('original_{}.png'.format(batch_idx),inputs[batch_idx][0])
#     fig.add_subplot(row, col, 2*batch_idx+2)
#     plt.imshow(outputs[batch_idx][0][0],cmap ='gray')
#     # plt.imsave('deepfool_{}.png'.format(batch_idx), outputs[batch_idx][0][0])
# plt.show()
## end of showing images ##

###### Writing pickles ######
out_file = "{}_{}.pkl".format(args.net,args.dataset)
print("==> Write pickle objects, file name ",out_file)
save_dictionary_to_pickle(outputs,out_file, image_channel, image_height, image_height)

###### end of pickle ######
'''
save_model(net, accuracy, oname, args.trial, epoch=0)
save_activations(file, 0, activs, targets)

for epoch in range(start_epoch, start_epoch+args.epochs):
    print('Epoch {}'.format(epoch))
    train(net, trainloader, device, optimizer, criterion)
    accuracy, activs, targets = test(net, testloader_array[0], device, criterion, n_test_batches=10)

    lr_scheduler.step(accuracy)

    if epoch<10:
        save_model(net, accuracy, oname, args.trial, epoch)
        save_activations(file, epoch, activs, targets)  
    elif epoch%args.save_every==0:
        save_model(net, accuracy, oname, args.trial, epoch)
        save_activations(file, epoch, activs, targets)

file.close()
'''

'''Some helper functions for PyTorch, including:
1;95;0c    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import torchvision
import numpy as np
import h5py
import errno
import os.path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    if device == 'cuda':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def prepare_mnist():
    print('===> Preparing data...')
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    
    trainset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform_train)
    if device == 'cuda':
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    testset = torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform_test)

    if device == 'cuda':
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    return trainloader, testloader

    
def prepare_cifar10():
    # Data
    print('===> Preparing data...')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    if device == 'cuda':
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    if device == 'cuda':
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    else:       
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader,  testloader


def prepare_tiny_imagenet():
    # Pay attention: imagenet you should download maually
    print('==> Preparing data...')
    transform_train = transforms.Compose([
        transforms.Resize(32), 
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))        
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.ImageFolder(root='./tiny-imagenet-200/train/', transform=transform_train)
    if device == 'cuda':
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    # validation: used only once
    complete_file  =  './tiny-imagenet-200/val/complete.txt'
    if not os.path.exists(complete_file):
        val_file = './tiny-imagenet-200/val/val_annotations.txt'
        data_dir = './tiny-imagenet-200/val/images'
        prepare_validation_imagenet(data_dir, val_file)
        f = open(complete_file,'w+')
        f.write('validation completed')

    # end after execuated once
    testset = torchvision.datasets.ImageFolder(root='./tiny-imagenet-200/val/images/', transform=transform_test)
    if device == 'cuda':
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)


    return trainloader,  testloader


def prepare_validation_imagenet(data_dir, validation_labels_file):
    
    # Read the synsets  associated with the validation data set.
    labels = [l.strip().split('\t')[1] for l in open(validation_labels_file).readlines()]
    unique_labels = set(labels)

    # Make all sub-directories in the validation data dir.
    for label in unique_labels:
        labeled_data_dir = os.path.join(data_dir, label)
        
        # Catch error if sub-directory exists
        try:
            os.makedirs(labeled_data_dir)
        except OSError as e:
            # Raise all errors but 'EEXIST'
            if e.errno != errno.EEXIST:
                raise

    # Move all of the image to the appropriate sub-directory.
    for i in range(len(labels)):
        basename = 'val_%d.JPEG' % i
        original_filename = os.path.join(data_dir, basename)
        if not os.path.exists(original_filename):
            print('Failed to find: %s' % original_filename)
            sys.exit(-1)
        new_filename = os.path.join(data_dir, labels[i], basename)    
        os.rename(original_filename, new_filename)
        
def train(net, trainloader, device, optimizer, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(net, testloader, device, criterion, n_test_batches):
    net.eval()
    test_loss, correct, total, target_acc, activation_acc = 0, 0, 0, [], []
    print("=> test data...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if batch_idx < n_test_batches:
                if device == 'cuda':
                    activations = [a.cpu().data.numpy().astype(np.float16) for a in net.module.forward_features(inputs)]
                else:
                    activations = [a.cpu().data.numpy().astype(np.float16) for a in net.forward_features(inputs)]
                target_acc.append(targets.cpu().data.numpy())
                activation_acc.append(activations)

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100.*correct/total
                 
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%%'
                     % (test_loss/(batch_idx+1), accuracy))
          

    activs = [np.concatenate(list(zip(*activation_acc))[i]) for i in range(len(activation_acc[0]))]
    
    return (accuracy, activs, np.concatenate(target_acc))


def save_model(net, acc, save_name, trial, epoch):
    print('Saving checkpoint...')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+save_name+'/ckpt_trial_'+str(trial)+'_epoch_'+str(epoch)+'.t7')

def save_activations(file, epoch, activs, targets):
    print('Saving activations...')
    for i, x in enumerate(activs):
        file.create_dataset("epoch_"+str(epoch)+"/activations/layer_"+str(i), data=x, dtype=np.float16)
    file.create_dataset("epoch_"+str(epoch)+"/targets", data=targets)

## Deal with deepfool ##
from deepfool import deepfool

# lightweight application of deepfool
def deep_fool_images(net, testloader, num_classes = 10):
    # input:    net -- input network
    #           testloader/testloader_array -- test images and targets set
    #           num_classes -- number of classes of the current dataset
    #           n_test_batches -- (not used yet) test_batches
    # output:   loader -- new data loader
    #           outputs -- [list] output images [list[i][0]] with targets [list[i][1]]
    # function: Apply deepfool with network to testloader
    # Note:     batch_size is 100 by default
    net.eval()
    print("==> Deep fool used for the dataset...")
    outputs = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if( batch_idx>=dataNum/100):
            break
        inputs, targets = inputs.to(device), targets.to(device)
        print("==> iteraion ", batch_idx)
        for i in range(100):
            [min_pert, loop_i, label, k_i, pert_image] = deepfool(inputs[i].to(device),net,num_classes)
            outputs.append([pert_image[0],targets[i]])


    loader = torch.utils.data.DataLoader(outputs, batch_size=100, shuffle=False)
    return loader,outputs

# lightweight test
def deep_fool_test(net,outputs,criterion):
    # output accuracy
    imgLen = len(outputs)
    total = 0
    net.eval()
    test_loss, correct, total, target_acc, activation_acc = 0, 0, 0, [], []

    for i in range(imgLen):
        img = np.reshape(outputs[i][0],[1,1,28,28])
        out = net(img)
        label = torch.max(out,1)[1]

        target = outputs[i][1]
        correct+=label.eq(target)
        print(target,label,correct)

        # loss = criterion(out, target)
        # test_loss += loss.item()
        # _, predicted = outputs.max(1)
        # correct += predicted.eq(target).sum().item()

    accuracy = correct * 100 / imgLen
    progress_bar(i, imgLen, 'Loss: %.3f | Acc: %.3f%%'
                     % (test_loss/(i+1), accuracy))
    return accuracy

# save or load with pickle
import pickle
dataNum = 1000 # Default number
def save_dictionary_to_pickle(outputs,file_name, channel, height, width):
    # function: Used to save a dictionary to pickle object
    # input:    [outputs]: should be a list with [image, label]
    #           [file_name]: the file name of pickle object (format *.pkl)
    #           # the following 3 params will change the dictionary structure of key'images'
    #           [channel]: the channel of the image ( 1 or 3 )
    #           [height]: the height of the image
    #           [width]: the width of the image
    # no return
    # Note: use binary format to save

    images = np.ndarray([dataNum,channel,height,width])
    labels = np.ndarray([dataNum,1])
    for i in range(dataNum):
        images[i,:,:,:] = outputs[i][0]
        labels[i,:] = outputs[i][1]

    print(images.shape,labels.shape)
    dir = {'images':images, 'labels':labels}
    out_file = open(file_name,'wb')
    pickle.dump(dir,out_file)
    out_file.close()

def load_dictionary_from_pickle(filename):
    # function: Used to load a list/loader from pickle object
    # input:    [file_name]: the file name of pickle object (format *.pkl)
    # output:   deepfoolLoader -- data loader with the deepfooled images
    #           outputs -- [list] output images [list[i][0]] with targets [list[i][1]] numpy.ndarray
    # change to cpu with a.cpu()
    pkl = pickle.load(open(filename,'rb'))
    imgs = pkl['images']
    lbls = pkl['labels']
    outputs = []

    for i in range(dataNum):
        img = torch.from_numpy(imgs[i]).float()
        img = img.cpu()
        lbl = torch.from_numpy(lbls[i]).long()[0]
        lbl = lbl.cpu()
        # print("image is {}, label is {}".format(img.shape,type(lbl)))
        # img, lbl = img.to(device), lbl.to(device)
        outputs.append([img,lbl])

    if device == 'cpu':
        deepfoolLoader = torch.utils.data.DataLoader(outputs, batch_size=10, shuffle=False)
    else:
        deepfoolLoader = torch.utils.data.DataLoader(outputs, batch_size=10, shuffle=False, num_workers = 2)
    return deepfoolLoader, outputs

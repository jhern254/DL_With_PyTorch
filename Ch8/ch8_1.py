import os
import numpy as np
import collections
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def set_path(directory):
    DEBUG = False
    #set wd
    path = ('/home/jun/Documents/Programming/DL_With_PyTorch/'
             + directory + '/')
    os.chdir(path)
    if(DEBUG):
        print(os.getcwd())


def ex1():
    print('\nEx1')
    PRINT_FIGS = False 

    # 3 rgb channel input, 16 tensor output
    conv = nn.Conv2d(3, 16, kernel_size=3) # 3x3 kernel
    print(conv)
# Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
    print(conv.weight.shape, conv.bias.shape)
# torch.Size([16, 3, 3, 3]) torch.Size([16])

    img, _ = cifar2[0] # 3x32x32 image, and class label
    output = conv(img.unsqueeze(0)) # conv expects b, B x C x W x H
    print(img.unsqueeze(0).shape, output.shape)
# torch.Size([1, 3, 32, 32]) torch.Size([1, 16, 30, 30])
# losing pixels around edges, need padding

    # print out grayscale img
    if(PRINT_FIGS):
        plt.clf()
        plt.cla()
        plt.figure(figsize=(10, 4.8))  # bookskip
        ax1 = plt.subplot(1, 2, 1)   # bookskip
        plt.title('output')   # bookskip
        plt.imshow(output[0, 0].detach(), cmap='gray')
        plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)  # bookskip
        plt.imshow(img.mean(0), cmap='gray')  # bookskip
        plt.title('input')  # bookskip
        plt.savefig('F1.png')  # bookskip

    # set up 2d conv. w/ padding
    conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
    output = conv(img.unsqueeze(0))
    print(img.unsqueeze(0).shape, output.shape)
# torch.Size([1, 3, 32, 32]) torch.Size([1, 1, 32, 32])
# Now, no missing pixels

    # Play w/ custom weights
    print('\nSec. 8.2.2 - Custom weights')

    # zero out bias tensor
    with torch.no_grad():
        conv.bias.zero_()
    # fill weights w/ avg for 3x3 kernel-Constant Conv. Kernel
    with torch.no_grad():
        conv.weight.fill_(1.0 / 9.0)
    # See effect on img
    output = conv(img.unsqueeze(0))
    plt.clf()
    plt.cla()
    plt.imshow(output[0, 0].detach(), cmap='gray')
    plt.savefig('8.2.2.png')

    # book plot of input and output
    if(PRINT_FIGS):
        plt.clf()
        plt.cla()
        output = conv(img.unsqueeze(0))
        plt.figure(figsize=(10, 4.8))
        ax1 = plt.subplot(1, 2, 1)
        plt.title('output')
        plt.imshow(output[0, 0].detach(), cmap='gray')
        plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
        plt.imshow(img.mean(0), cmap='gray')
        plt.title('input')
        plt.savefig('F4_ConstantConvKernel.png')
# Image looks blurry, bird outline kind of lost
# Every pixel of output is avg of a neighborhood of the input of pixels
# so pixels in output are correlated and change more smoothly

    # try Edge-Detection kernel
    conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
    with torch.no_grad():
        conv.weight[:] = torch.tensor([[-1.0, 0.0, 1.0],
                                       [-1.0, 0.0, 1.0],
                                       [-1.0, 0.0, 1.0]])
        # set bias to zero
        conv.bias.zero_()

    if(PRINT_FIGS):
        plt.clf()
        plt.cla()
        output = conv(img.unsqueeze(0))
        plt.figure(figsize=(10, 4.8))  # bookskip
        ax1 = plt.subplot(1, 2, 1)   # bookskip
        plt.title('output')   # bookskip
        plt.imshow(output[0, 0].detach(), cmap='gray')
        plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)  # bookskip
        plt.imshow(img.mean(0), cmap='gray')  # bookskip
        plt.title('input')  # bookskip
        plt.savefig('F5_EdgeDetectionKernel.png')  # bookskip
        plt.show()

###############################
# Sec. 8.2.3
    print('\nMax Pool Downsampling:')

    pool = nn.MaxPool2d(2)
    output = pool(img.unsqueeze(0))
    print(img.unsqueeze(0).shape, output.shape)
#Max Pool Downsampling:
#torch.Size([1, 3, 32, 32]) torch.Size([1, 3, 16, 16])

    print('\nEx1 end')


def ex2():
    # Sec. 8.2.4
    print('\nEx2 start')
    # Ex. fully connected model
    model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.MaxPool2d(2),
                # ... Need to turn into 1d vector
                )

    model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.Tanh(), 
                nn.MaxPool2d(2),
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.MaxPool2d(2),
                # ...<1> Something important missing!
                nn.Linear(8 * 8 * 8, 32),
                nn.Tanh(),
                nn.Linear(32, 2))
    
    numel_list = [p.numel() for p in model.parameters()]
    print(sum(numel_list), numel_list)
#18090 [432, 16, 1152, 8, 16384, 32, 64, 2]
# total params, params per layer

    # throws error
#    model(img.unsqueeze(0))

    print('\nEx2 end')

# base CNN network submodule 
class Net(nn.Module):
    def __init__(self):
        super().__init__() # need to call super
        # define layers w/ pytorch fns
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 2)

# explicitly written out forward fn, not limited by nn.Sequential
    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x))) # layer 1
        out = self.pool2(self.act2(self.conv2(out))) # layer 2
        # reshape for linear layer
        out = out.view(-1, 8 * 8 * 8) # B x N tensor, B is any n
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out




if __name__ == "__main__":
#############################################3##################################
#############################################3##################################
########## Set up dataset
    torch.manual_seed(123)  # from notebook

    set_path('Ch8')
    data_path = os.getcwd()
#    print(data_path)

    class_names = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

    # import transformed dataset w/ transform arg, and Compose
    transformed_cifar10 = datasets.CIFAR10(
        data_path, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
        ]))
    # import transformed val. dataset w/ transform arg, and Compose
    transformed_cifar10_val = datasets.CIFAR10(
        data_path, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
        ]))

# Subsetting data to airplane and bird images only
    # filter data in cifar10 and remap the labels so they're contiguous
    label_map = {0: 0, 2: 1}
    class_names = ['airplane', 'bird']
    # create subset for airplane and bird
    cifar2 = [(img, label_map[label])
               for img, label in transformed_cifar10
               if label in [0, 2]]
    cifar2_val = [(img, label_map[label])
                   for img, label in transformed_cifar10_val
                   if label in[0, 2]]

    print(len(cifar2), type(cifar2), type(cifar2[0]))
#############################################3##################################
#############################################3##################################

#    ex1()        
    ex2()

    print('\nEnd main\n')



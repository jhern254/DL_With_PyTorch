import os
import numpy as np
import collections
import logging
import datetime
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import NetDepth

logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.DEBUG)

def set_path(directory):
    DEBUG = False
    #set wd
    path = ('/home/jun/Documents/Programming/DL_With_PyTorch/'
             + directory + '/')
    os.chdir(path)
    if(DEBUG):
        print(os.getcwd())

# base CNN
class Net1(nn.Module):
    '''
        input - B x 3C x W x H Tensor
    '''
    def __init__(self):
        super().__init__() # init. super nn class
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 2)

# at minimum, need to implement forward pass
    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))   # layer 1
        out = self.pool2(self.act2(self.conv2(out))) # layer 2
        # reshape tensor for linear layer
        out = out.view(-1, 8 * 8 * 8)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out      # return tensor


class Net2(nn.Module):
    """
        Using functional F module since tanh and maxpool2d only return values
        Output:
            DEBUG:x is torch.Size([1, 3, 32, 32])
            DEBUG:out1 is torch.Size([1, 16, 16, 16])
            DEBUG:out2 is torch.Size([1, 8, 8, 8])
            DEBUG:out3 is torch.Size([1, 512])
            DEBUG:out4 is torch.Size([1, 32])
            DEBUG:returning out: torch.Size([1, 2])
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        logging.debug('x is %s', x.shape)
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        logging.debug('out1 is %s', out.shape)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        logging.debug('out2 is %s', out.shape)
        out = out.view(-1, 8 * 8 * 8)
        logging.debug('out3 is %s', out.shape)
        out = torch.tanh(self.fc1(out))
        logging.debug('out4 is %s', out.shape)
        out = self.fc2(out)
        logging.debug('returning out: %s', out.shape)
        return out



class NetWidth(nn.Module):
    def __init__(self, n_chans=32):
        """ Input tensor B x C x W x H
        """
        # must call super for init
        super().__init__()
        self.n_chans = n_chans
        self.conv1 = nn.Conv2d(3, n_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans, n_chans // 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * n_chans // 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # conv layers
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)     # conv1(x) >> tanh >> maxpool()
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        # linear layers(regression)
        out = out.view(-1, 8 * 8 * self.n_chans // 2) # 1 x N dim. vector/reg.
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


class NetDropout(nn.Module):
    def __init__(self, n_chans=32):
        super().__init__()
        self.n_chans = n_chans
        self.conv1 = nn.Conv2d(3, n_chans, kernel_size=3, padding=1)
        self.conv1_dropout = nn.Dropout2d(p=0.4)
        self.conv2 = nn.Conv2d(n_chans, n_chans // 2, kernel_size=3, padding=1)
        self.conv2_dropout = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(8 * 8 * n_chans // 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = self.conv1_dropout(out)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = self.conv2_dropout(out)
        out = out.view(-1, 8 * 8 * self.n_chans // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out




def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):    # don't forget +1
        loss_train = 0.0
        for imgs, labels in train_loader:
            # set up move to gpu
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            # train
            outputs = model(imgs)
            # loss
            loss = loss_fn(outputs, labels)

            # zero gradients
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # step forward/ update params
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)
                ))

def training_loop_l2reg(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            # move to device
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            # train
            outputs = model(imgs)
            # loss
            loss = loss_fn(outputs, labels)

            #L2 Regulization(Ridge regression)
            l2_lambda = 0.001
            # sum of squares
            l2_norm = sum(p.pow(2.0).sum()
                            for p in model.parameters())
            loss = loss + l2_lambda * l2_norm

            # zero gradients
            optimizer.zero_grad()
            # back prop
            loss.backward()
            # update weights
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))





def validate1(model, train_loader, val_loader):
    accdict = {}
    # based on what loader returns
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        # check current weights performance
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                # find max prob. in 1st dim/prob. values
                _, predicted = torch.max(outputs, dim=1)
                # find accuracy
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name, correct / total))
        accdict[name] = correct / total
    return accdict
# OrderedDict([('width', {'train': 0.9649, 'val': 0.8895})])











if __name__ == "__main__":
## Set up dataset
    torch.manual_seed(123)  # from notebook

    set_path('Testing')
    data_path = os.getcwd()
    print(data_path)

    class_names = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

    # import transformed dataset w/ transform arg, and Compose
    transformed_cifar10 = datasets.CIFAR10(
        data_path, train=True, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
        ]))
    # import transformed val. dataset w/ transform arg, and Compose
    transformed_cifar10_val = datasets.CIFAR10(
        data_path, train=False, download=False,
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

########################################################################
# Examples
    img, _ = cifar2[0] # 3x32x32 image, and class label
    print(img.shape, img.unsqueeze(0).shape)
#torch.Size([3, 32, 32]) torch.Size([1, 3, 32, 32])

    all_acc_dict = collections.OrderedDict()
    # use gpu
    device = (torch.device('cuda') if torch.cuda.is_available()
                else torch.device('cpu'))
    print(f"Training on device {device}.")

    """ Net1 """
    if(False):
        model = Net1()
        numel_list = [p.numel() for p in model.parameters()]
#        print(sum(numel_list), numel_list)
# 18090 [432, 16, 1152, 8, 16384, 32, 64, 2]
        test = model(img.unsqueeze(0))
        print(test)
# tensor([[0.0909, 0.0939]], grad_fn=<AddmmBackward>)

    """ Net2 """
    if(False):
        model = Net2()
        numel_list = [p.numel() for p in model.parameters()]
        print(sum(numel_list), numel_list)
# 18090 [432, 16, 1152, 8, 16384, 32, 64, 2]
        test = model(img.unsqueeze(0))
        print(test)
#DEBUG:x is torch.Size([1, 3, 32, 32])
#DEBUG:out1 is torch.Size([1, 16, 16, 16])
#DEBUG:out2 is torch.Size([1, 8, 8, 8])
#DEBUG:out3 is torch.Size([1, 512])
#DEBUG:out4 is torch.Size([1, 32])
#DEBUG:returning out: torch.Size([1, 2])
# tensor([[0.0909, 0.0939]], grad_fn=<AddmmBackward>)

# load data
    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                                shuffle=False)
    val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64,
                                                shuffle=False)

    """ NetWidth """
# test new training loop and NetWidth network
    if(False):
        # works w/ .toGPU since its a subclass of nn.Module
        model = NetWidth().to(device=device)
        n_epochs = 100
        # tune learning rate hyper param.
        optimizer = optim.SGD(model.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        training_loop(
                n_epochs = n_epochs,
                optimizer = optimizer,
                model = model,
                loss_fn = loss_fn,
                train_loader = train_loader
                )
        all_acc_dict["width"] = validate1(model, train_loader, val_loader)
        print(sum(p.numel() for p in model.parameters()))
#Accuracy train: 0.96
#Accuracy val: 0.89
#OrderedDict([('width', {'train': 0.9649, 'val': 0.8895})])
#38386

    """ Training Loop L2 Reg """
    if(False):
        model = NetWidth().to(device=device)
        optimizer = optim.SGD(model.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        training_loop_l2reg(
            n_epochs = 100,
            optimizer = optimizer,
            model = model,
            loss_fn = loss_fn,
            train_loader = train_loader
            )
        all_acc_dict["l2 reg"] = validate1(model, train_loader, val_loader)
#Accuracy train: 0.95
#Accuracy val: 0.89
#OrderedDict([('width', {'train': 0.9649, 'val': 0.8895}), 
#('l2 reg', {'train': 0.9497, 'val': 0.89})])

    """ NetDropout """
    if(False):
        model = NetDropout().to(device=device)
        optimizer = optim.SGD(model.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        training_loop(
                n_epochs = 100,
                optimizer = optimizer,
                model = model,
                loss_fn = loss_fn,
                train_loader = train_loader
            )
        all_acc_dict["dropout"] = validate1(model, train_loader, val_loader)
        
#    print(all_acc_dict)

    
#    NetDepth.SayHello()
    if(True):
        model = NetDepth.NetDepth().to(device=device)
        optimizer = optim.SGD(model.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        training_loop(
                n_epochs = 100,
                optimizer = optimizer,
                model = model,
                loss_fn = loss_fn,
                train_loader = train_loader
            )
        all_acc_dict["NetDepth"] = validate1(model, train_loader, val_loader)

        # TODO: how to print final vector?
        print(model)

    print(all_acc_dict)


#    logging.debug('Test is %s',test)
    print('\nEnd main\n')








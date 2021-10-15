import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from matplotlib import pyplot as plt
from torchvision import transforms


def test():
    print(temp)


def set_path(directory):
    DEBUG = False 
    #set wd
    path = ('/home/jun/Documents/Programming/DL_With_PyTorch/'
             + directory + '/')
    os.chdir(path)
    if(DEBUG):
        print(os.getcwd())


def softmax_fn(x):
    return torch.exp(x) / torch.exp(x).sum()


def ex1():
    print('\nStart ex. 1')
    print(dir(transforms))

    # fn returns tensor from PIL img input
    to_tensor = transforms.ToTensor()
    img_t = to_tensor(img)
    print(img_t.shape)  # torch.Size([3, 32, 32])
    # data loader w/ transforms arg for tensor
    tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=False,
                                      transform=transforms.ToTensor())
    print(type(tensor_cifar10))
    print(len(tensor_cifar10))
#<class 'torchvision.datasets.cifar.CIFAR10'>
#50000

    # _ var is ignore value or use as placeholder(idx w/out name)
    img_t, _ = tensor_cifar10[99]
    print(type(img_t), img_t.shape, img_t.dtype, type(_), _) 
# <class 'torch.Tensor'> torch.Size([3, 32, 32]) torch.float32 <class 'int'> 1
# IMPORTANT ##########
# The way python works is that multi values get assigned here, the value(img tensor)
# and the index

    # since img is tensor now, values are between (0,1), instead of (0, 255) as image
    print(img_t.min(), img_t.max())
# tensor(0.) tensor(1.)

    # Verify we are getting same image
    # have to permute  from C x H x W to H x W x C for Matplotlib
    plt.clf()   # Reset prev. plt
    plt.cla()
    plt.imshow(img_t.permute(1, 2, 0))
    plt.savefig('7.4.png')
    print(img_t.shape) # torch.Size([3, 32, 32]), still same shape

#######################
#######################
# Ex. 7.1.4 Normalizing Data
    print('\nEx.7.1.4 Normalizing Data')
    # stack all Cifar10 images along 4th dim
    imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)
    print(imgs.shape)   # torch.Size([3, 32, 32, 50000])

    # Compute mean per channel
    # See book: View keeps dim 0, merges rest(-1), then avgs.
    print(imgs.view(3, -1).mean(dim=1)) # imgs does not change orig. tensor
# tensor([0.4914, 0.4822, 0.4465])

    # compute std dev
    print(imgs.view(3, -1).std(dim=1))
# tensor([0.2470, 0.2435, 0.2616])

    # Normalize based on mean and std
    print(transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616)))
# Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616))

    # import transformed dataset w/ transform arg, and Compose
    transformed_cifar10 = datasets.CIFAR10(
        data_path, train=True, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
        ]))

    img_t, _ = transformed_cifar10[99]
    print(type(img_t))
    plt.clf()
    plt.cla()
    plt.imshow(img_t.permute(1, 2, 0))
    plt.savefig('7.5.png')
    
    # import transformed val. dataset w/ transform arg, and Compose
    transformed_cifar10_val = datasets.CIFAR10(
        data_path, train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
        ]))

    print('\nEnd ex. 1')


def ex2():
    print('\nEx2')
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
    if(True):
        print(len(transformed_cifar10)) # 50,000
#        print(cifar2[0])
#(tensor([[[ 0.6143, -0.3224, -0.1160,  ..., -0.2589, -0.2748, -0.5447],
#         [ 0.6620, -0.1478, -0.8463,  ..., -0.3224, -0.3224, -0.5764],
#         [ 0.2333,  0.2650, -0.1001,  ..., -0.3383, -0.6558, -0.7511],
#         ...,
#         [ 0.2174,  0.2650,  0.1539,  ..., -0.5764, -0.4494,  0.0110],
#         [ 0.5984,  0.4397,  0.3285,  ..., -0.6399, -0.4335,  0.0269],
#         [ 0.9160,  0.8048,  0.4556,  ..., -0.4971, -0.5447, -0.0525]],
#
#        [[ 1.3373,  0.2744,  0.4033,  ...,  0.3871,  0.3871,  0.0973],
#         [ 1.4501,  0.5965, -0.2248,  ...,  0.3066,  0.3066,  0.0650],
#         [ 1.0958,  1.1280,  0.6448,  ...,  0.2583, -0.0477, -0.1282],
#         ...,
#         [ 0.4033,  0.5160,  0.5321,  ...,  0.1778,  0.4033,  0.8542],
#         [ 0.5482,  0.6609,  0.6609,  ...,  0.1134,  0.4033,  0.8864],
#         [ 0.4838,  0.9508,  0.4999,  ...,  0.1778,  0.1617,  0.7576]],
#
#        [[-0.4476, -0.7924, -0.1927,  ..., -0.6125, -0.6724, -0.8523],
#         [-0.4476, -0.9723, -1.0622,  ..., -0.5225, -0.6275, -0.8523],
#         [-0.7324, -0.7174, -0.5225,  ..., -0.4476, -0.8373, -0.9723],
#         ...,
#         [-0.4926, -0.5975, -0.6275,  ..., -1.2871, -1.3470, -0.9723],
#         [-0.4326, -0.4776, -0.3576,  ..., -1.4220, -1.3021, -0.9873],
#         [-0.1778,  0.0321, -0.2077,  ..., -1.2721, -1.3170, -1.0472]]]), 1)
# Tuple obj, first stored value is image tensor, 2nd value is class label

        print(len(cifar2), type(cifar2), type(cifar2[0]))
# 10000 <class 'list'> <class 'tuple'>. This is a list of tuples

#        print(cifar2[0].dtype) # Doesn't work since tuple of tens. and idx var
        # have to assign mult. vars to access
        ten, idx = cifar2[0]
        print(ten.dtype, ten.shape, idx) # torch.float32 1
# torch.float32 torch.Size([3, 32, 32]) 1

        print('\n\n\nDEBUG')


    # Ex model - 3 x 32 x 32 = 3072 input features
    n_out = 2
    model = nn.Sequential(
                nn.Linear(
                    3072,
                    512, # arbitrary
                ),
                nn.Tanh(),
                nn.Linear(
                    512,
                    n_out,
                )
    )
    print(model)
    # test out softmax fn
    x = torch.tensor([1.0, 2.0, 3.0])
    print(softmax_fn(x)) # returns prob.
# tensor([0.0900, 0.2447, 0.6652])
    print(softmax_fn(x).sum()) # tensor(1.)

    # nn softmax fn, requires dim arg
    softmax = nn.Softmax(dim=1)
    x = torch.tensor([[1.0, 2.0, 3.0],
                     [1.0, 2.0, 3.0]])
    print(softmax(x))
#tensor([[0.0900, 0.2447, 0.6652],
#        [0.0900, 0.2447, 0.6652]])

    # real model
    model = nn.Sequential(
                nn.Linear(3072, 512),
                nn.Tanh(),
                nn.Linear(512, 2),
                nn.Softmax(dim=1)
    )
    # test out model before using
    plt.clf()
    plt.cla()
    img, _ = cifar2[0]
    print(type(img))
    plt.imshow(img.permute(1, 2, 0))
    plt.savefig('7.9.png')  
    
    # compress all dims, add batch dim - 1 x 3,072
    img_batch = img.view(-1).unsqueeze(0)
    print(img_batch.shape) # torch.Size([1, 3072])
    # run model
    out = model(img_batch)
    print(out)
# tensor([[0.4471, 0.5529]], grad_fn=<SoftmaxBackward>)
# model gives 55% prob. that image is bird - bad model, weights are garb. values

    # prints max prob. dim
    _, index = torch.max(out, dim=1)
    print(index) 
# tensor([1])

    print('\nEx2 end')


def ex3():
    print('\nEx3')
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

#######################
#######################
# 7.2.5
    # NLL/MLE loss for model 
    model = nn.Sequential(
                nn.Linear(3072, 512),
                nn.Tanh(),
                nn.Linear(512, 2),
                nn.LogSoftmax(dim=1)) # need log for NLL
    loss = nn.NLLLoss()
    
    img, label = cifar2[0]
    out = model(img.view(-1).unsqueeze(0)) # 2 dim
    print(out)
    # compute loss
    print(loss(out, torch.tensor([label])))
#tensor([[-0.8798, -0.5359]], grad_fn=<LogSoftmaxBackward>)
#tensor(0.5359, grad_fn=<NllLossBackward>)

    print(len(cifar2)) # 10000

    # 7.2.6 - train classifier model
    model = nn.Sequential(
                nn.Linear(3072, 512),
                nn.Tanh(),
                nn.Linear(512, 2),
                nn.LogSoftmax(dim=1)) # need log for NLL
    learning_rate = 1e-2
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()
    n_epochs = 10
    # training loop - updates params. after checking EVERY sample indiv. 
    if(False):
        for epoch in range(n_epochs):
            for img, label in cifar2:
                # train model and comput loss
                out = model(img.view(-1).unsqueeze(0)) # squish to one value, add batch
                loss = loss_fn(out, torch.tensor([label]))
                # zero grad, backprop, step params
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print epoch info
            print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
    # This trains really slow, batch training is better.

    # Set up minibatches
    # Use dataloader to set up minibatches
    print('\nTrain set:')
    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                               shuffle=True)
    model = nn.Sequential(
                nn.Linear(3072, 512),
                nn.Tanh(),
                nn.Linear(512, 2),
                nn.LogSoftmax(dim=1)) # need log for NLL
    learning_rate = 1e-2
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()
    n_epochs = 100

    DEBUGL = False 
    # training loop w/ batches of size 64
    if(False):
        for epoch in range(n_epochs):
            if(DEBUGL):
                idx = 0
            for imgs, labels in train_loader:
#                print(imgs.shape) # torch.Size([64, 3, 32, 32])
                # set batch size of images to train on, then model on that
                batch_size = imgs.shape[0] # 64
                outputs = model(imgs.view(batch_size, -1)) # [64, 3,072], no unsq.
                loss = loss_fn(outputs, labels)
                # zero grad., backprop, step params
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if(DEBUGL):
                    print("Output Size:", outputs.shape)
                    print('Loop:', idx+1)
                    idx += 1
#Output Size: torch.Size([64, 2])
#Loop: 155
#Output Size: torch.Size([64, 2])
#Loop: 156
#Output Size: torch.Size([16, 2])
#Loop: 157
#Epoch: 2, Loss: 0.514087
# LOGIC ############################
# This makes sense. Each loop model outputs tens. of 64x2, until last output 16x2
# 64 * 156 = 9984 + 16 = 10K images
# I'm guessing that the outer epoch loop is able to randomize(permute) so that
# each epoch works w/ the 10k images permutated

            # print batch info
            print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
#            print("Output Size:", outputs.shape)
# Output Size: torch.Size([16, 2]). Prints after every loop
        print(outputs, outputs.shape)
#Epoch: 0, Loss: 0.654031
#Epoch: 1, Loss: 0.309752
#Epoch: 2, Loss: 0.460587
#Epoch: 3, Loss: 0.572007
#...
#Epoch: 96, Loss: 0.018994
#Epoch: 97, Loss: 0.025595
#Epoch: 98, Loss: 0.032078
#Epoch: 99, Loss: 0.012303
# Loss went down on training, updated params. every batch. Loss went down after ep. 50
# looks good. Now to test on valid. set.

#tensor([[-4.4751e-03, -5.4115e+00],
#        [-6.8083e+00, -1.1052e-03],
#        [-6.0576e-04, -7.4094e+00],
#        [-2.0807e+00, -1.3335e-01],
#        [-4.2768e+00, -1.3985e-02],
#        [-1.5285e-02, -4.1885e+00],
#        [-4.4312e+00, -1.1972e-02],
#        [-5.9455e+00, -2.6210e-03],
#        [-3.9963e+00, -1.8554e-02],
#        [-5.0121e+00, -6.6795e-03],
#        [-1.0018e-01, -2.3504e+00],
#        [-3.8763e+00, -2.0946e-02],
#        [-3.7860e-02, -3.2927e+00],
#        [-5.9604e-06, -1.2038e+01],
#        [-8.1121e-03, -4.8185e+00],
#        [-4.3109e-02, -3.1655e+00]], grad_fn=<LogSoftmaxBackward>) torch.Size([16, 2])
# Is this correct? See dims.
# Ans: Yes, see prev. This is last batch of imgs output by model
# Check params tensor

    # Post training training acc. check
    print('\nTrain set acc. check:')
    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                               shuffle=False)
    # make correct and total vars to keep track in loop
    correct = 0
    total = 0
    if(False):
        with torch.no_grad(): # no grad for acc. check
            for imgs, labels in train_loader:
                outputs = model(imgs.view(imgs.shape[0], -1))
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print("Accuracy: %f" % (correct / total))
        print('Total: ', total, ' Correct: ', correct)

#Train set acc. check:
#Accuracy: 0.999000
#Total:  10000  Correct:  9990

    # Test model on valid. set
    # load valid set
    print('\nValid. Set:')
    val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64,
                                             shuffle=False)
    # make correct and total vars to keep track in loop
    correct = 0
    total = 0
    if(False):
        with torch.no_grad(): # no grad for val. set
            for imgs, labels in val_loader:
                # set batch size and train model, loss
                batch_size = imgs.shape[0] # batch size = 64
#                print(imgs.shape) # torch.Size([64, 3, 32, 32])
                outputs = model(imgs.view(batch_size, -1))
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print("Accuracy: %f" % (correct / total))
        print('Total: ', total, ' Correct: ', correct)
#Valid. Set:
#Accuracy: 0.815500
# Total:  2000  Correct:  1631

#######################
#######################
# New Model

    print('\nNew Model w/ CrossEntropyLoss & 3 Hidden layers:')
    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                               shuffle=True) # shuffle is T now

    model = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(), 
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
#            nn.LogSoftmax() # Can use CrossEntropyL = Softmax + NLLL output
    ) 
    # CrossEntropyLoss in PyTorch = LogSoftmax + NLLL
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-2
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    n_epochs = 100
    
    # training loop
    if(True):
        for epoch in range(n_epochs):
            # imgs and labels are what they are, an img and class label
            for imgs, labels in train_loader:
                # set batch
                batch_size = imgs.shape[0]
                # calc train model and loss
                outputs = model(imgs.view(batch_size, -1))
                loss = loss_fn(outputs, labels)
                # zero grad., backprop, step params
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                
            # print epoch info
            print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
#Epoch: 96, Loss: 0.000585
#Epoch: 97, Loss: 0.000711
#Epoch: 98, Loss: 0.000749
#Epoch: 99, Loss: 0.000380
# Very small loss around epoch 70.

    # Post training training acc. check
    print('\nTrain set acc. check:')
    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                               shuffle=False)
    # make correct and total vars to keep track in loop
    correct = 0
    total = 0
    if(True):
        with torch.no_grad(): # no grad for acc. check
            for imgs, labels in train_loader:
                outputs = model(imgs.view(imgs.shape[0], -1))
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print("Accuracy: %f" % (correct / total))
        print('Total: ', total, ' Correct: ', correct)
#Train set acc. check:
#Accuracy: 1.000000
#Total:  10000  Correct:  10000

    # Test model on valid. set
    # load valid set
    print('\nValid. Set:')
    val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64,
                                             shuffle=False)
    # make correct and total vars to keep track in loop
    correct = 0
    total = 0
    if(True):
        with torch.no_grad(): # no grad for val. set
            for imgs, labels in val_loader:
                # set batch size and train model, loss
                batch_size = imgs.shape[0] # batch size = 64
                outputs = model(imgs.view(batch_size, -1))
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print("Accuracy: %f" % (correct / total))
        print('Total: ', total, ' Correct: ', correct)
#Valid. Set:
#Accuracy: 0.797500
#Total:  2000  Correct:  1595
# Actually worse performance on valid. set. Interesting.
# Also, since training acc. is 100% and valid. is not much better, this is overfitting 

    # Print # of model params.
    print('\nPrinting # of params in model:')
    # number of total params.
    numel_list = [p.numel()
                  for p in model.parameters()]
    print(sum(numel_list), numel_list) 
#Printing # of params in model:
#3737474 [3145728, 1024, 524288, 512, 65536, 128, 256, 2]

    # number of trainable params.
    numel_list = [p.numel()
                  for p in model.parameters()
                  if p.requires_grad == True]
    print(sum(numel_list), numel_list) 
#3737474 [3145728, 1024, 524288, 512, 65536, 128, 256, 2]

    # check first simple model # of params
    first_model = nn.Sequential(
                    nn.Linear(3072, 512),
                    nn.Tanh(),
                    nn.Linear(512, 2),
                    nn.LogSoftmax(dim=1)
    )
    numel_list = [p.numel() for p in first_model.parameters()]
    print(sum(numel_list), numel_list)
# 1574402 [1572864, 512, 1024, 2]

    # print specific layer params
    print('\nPrinting layer params:')
    print(sum([p.numel() for p in nn.Linear(3072, 512).parameters()]))
    print(sum([p.numel() for p in nn.Linear(3072, 1024).parameters()]))
#Printing layer params:
#1573376
#3146752
# GOOD PRACTICE: ***********************
# These line up w/ prev. output
# These linear comb. layers have A LOT of params

    # verify number of params. in lin. comb/ regression of weights and bias tensor
    linear = nn.Linear(3072, 1024)
    print(linear.weight.shape, linear.bias.shape)
#torch.Size([1024, 3072]) torch.Size([1024])

    print('\nEx3 end')


#######################
#######################
# Test functions
def temp_test1():
    temp_1 = 1

def temp_test2():
    print(temp_1)
#######################
#######################


if __name__ == "__main__":
    torch.manual_seed(28)
    set_path('Ch7') 
    data_path = os.getcwd()
#    print(data_path)

    # Get data
    # tried to make fn, didn't work
    cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
    cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)
    # set up stuff
    class_names = ['airplane','automobile','bird','cat','deer',
                    'dog','frog','horse','ship','truck']
    fig = plt.figure(figsize=(8,3))
    num_classes = 10
    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        ax.set_title(class_names[i])
        img = next(img for img, label in cifar10 if label == i)
        plt.imshow(img)
    plt.savefig('test')
# Study how this code works. Got from python notebook

    print(type(cifar10).__mro__)
    print(len(cifar10)) # 50000
    img, label = cifar10[99]
    print(img, label, class_names[label])
#<PIL.Image.Image image mode=RGB size=32x32 at 0x7FB0D32071C0> 1 automobile

    # doesn't work in terminal?
#    plt.imshow(img)
    img.save('car.jpg') # save image using PIL library. Savefig is better

#######################
#######################
# Test to see if functions inherit main vars - Yes I guess. This works
#    temp = "Hello"
#    test()
#######################
#######################
# Start ex1

#    ex1()
#    ex2()    
    ex3()


# test code
#    temp_test1()
#    temp_test2()


# Done
    print('\nEnd main')



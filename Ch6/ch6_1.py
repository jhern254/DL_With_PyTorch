import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from matplotlib import pyplot as plt

"""
IMPORTANT: Doing Insert(paste) in vim inserts tabs instead of spaces. Be careful!
"""


def ex1_debug():
    DEBUG1 = False
    DEBUG2 = False

    print('\nex1 debug')
    # set model as variable
    linear_model = nn.Linear(1, 1) # nn.Modules have def. ctors using __call__
    if(DEBUG1):
        print(linear_model.weight)

    temp = linear_model(t_un_val)

#    linear_model(t_un_val)
    print(linear_model)
    print(temp)
#Linear(in_features=1, out_features=1, bias=True)
#tensor([[5.1171],
#        [3.0119]], grad_fn=<AddmmBackward>)

    # print model class vars 
    print('\nPrinting model attributes')
    print(linear_model.weight)
    print(linear_model.type)
    print(temp.dtype)
    print(temp.shape)
    print(temp[:,0])
#Printing model attributes
#Parameter containing:
#tensor([[0.6284]], requires_grad=True)
#<bound method Module.type of Linear(in_features=1, out_features=1, bias=True)>
#torch.float32
#torch.Size([2, 1])
#tensor([5.1171, 3.0119], grad_fn=<SelectBackward>)

# temp is a tensor, so model weights are stored inside tensor obj. This can be
# hard to see if .weight attr. needed

    if(DEBUG2):
        print('\nPrinting class obj attr')
        print(dir(temp))
        print('\nCompare:')
        print(dir(linear_model(t_un_val)))
        print('end ex1 debug')
        # These are same obj!

def ex1():
    DEBUG1 = False
    DEBUG2 = False
    DEBUG3 = False

    print('\nEx1')

    # set model as variable- args(input size, output size, bias == True)
    linear_model = nn.Linear(1, 1) # nn.Modules have def. ctors using __call__
    if(DEBUG1):
        print(linear_model.weight)  # This is initialized to garbage values
        print(linear_model.bias)
        print('DEBUG1 END\n')

    linear_model(t_un_val)
    print(linear_model(t_un_val))	# This is a tensor
    print(linear_model(t_un_val).dtype)
    print(linear_model(t_un_val).shape)
#tensor([[5.1171],
#        [3.0119]], grad_fn=<AddmmBackward>)
#torch.float32
#torch.Size([2, 1])

    # print model class vars/attributes
    print(linear_model.weight)
    print(linear_model.bias)
#Parameter containing:
#tensor([[0.6284]], requires_grad=True)
#Parameter containing:
#tensor([-0.0296], requires_grad=True)

    if(DEBUG2):
        print('\nTesting if temp == linear_model(t_un_val)')
        temp = linear_model(t_un_val)
        print(temp)
        print(temp.dtype)
        print(temp.shape)
        print(linear_model.weight)
        print(linear_model.bias)
        print('DEBUG CODE END')
# THIS IS SAME OBJ, but how do I access the weights? 
# ANS: Tensor obj. has no weights, only model obj

# This initialization has garbage values for weights and bias, probably because
# there is no step function based on gradient.


#########################
#########################
    # torch ones example - wrong dimensions for linear model
    print('\nTorch ones bad example')
    x = torch.ones(1)   # tensor([1.])
    linear_model(x)     # Wrong dim. as input, PyTorch fills dim. 
    print(linear_model(x))

    # torch ones example
    print('\nTorch ones good example')
    x = torch.ones(10, 1)   # tensor([1.])
    linear_model(x)
    print(linear_model(x))
    print(linear_model.weight)
    print(linear_model.bias)

# DEBUG example
    if(DEBUG3):
        print('\nTEST CODE')
        y = torch.ones(10, 1)
        test = linear_model(y)
        print(test)
        print(linear_model.weight)
        print(linear_model.bias)

        # doesn't work since linear_model returns tensor obj
        #print(linear_model(y).weight)
        #print(linear_model(y).bias)
        print(test.dtype)
        print('\nTEST CODE END')

# This all prints the same weight and bias. I think its because it's set to default,
# but then are the values garbage values? How do I initialize? Run test code
# ANS: After testing, I believe these are all garbage values, w/ no update step
# So, I think linear_model var is mutable, and the update step for optimizer
# will change the params inside obj.

    print('\nEx1 End')


def test():
    print('\n\nTEST FUNCTION')
    linear_model2 = nn.Linear(1, 1)
    print(linear_model2, linear_model2.weight, linear_model2.bias)
    print('\n\nTEST FUNCTION END')


def test2():
    print('\n\nTEST FUNCTION')
    linear_model3 = nn.Linear(1, 1)
    print(linear_model3, linear_model3.weight, linear_model3.bias)
    print('\n\nTEST FUNCTION END')


#########################
#########################
def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val,
                  t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):
        # train training model and loss
        t_pred_train = model(t_u_train) # no model.parameters() ?
        loss_train = loss_fn(t_pred_train, t_c_train)
        # train valid. model and loss
        t_pred_val = model(t_u_val)
        loss_val = loss_fn(t_pred_val, t_c_val)

        # zero grad., backprop, update params.
        optimizer.zero_grad()
        loss_train.backward()   # have to pass in loss fn
        optimizer.step()

        # print epoch info
        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                  f" Validation loss {loss_val.item():.4f}")


def ex2():
    print('\nEx2')
    linear_model = nn.Linear(1, 1)
    optimizer = optim.SGD(
        linear_model.parameters(),  # same as [params] from prev. ch.
        lr=1e-2
    )

    # needs to be list of tensors to print    
    print(list(linear_model.parameters()))  
#[Parameter containing:
#tensor([[0.6284]], requires_grad=True), Parameter containing:
#tensor([-0.0296], requires_grad=True)]

#########################
#########################
    print('\nTraining Loop ex:')
    linear_model = nn.Linear(1, 1)
    optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)
    # Yup, parameters tensor is mutable

    tloop = training_loop(
        n_epochs = 3000,
        optimizer = optimizer,
        model = linear_model,
        loss_fn = nn.MSELoss(),
        t_u_train = t_un_train,
        t_u_val = t_un_val,
        t_c_train = t_c_train,
        t_c_val = t_c_val
        )
#Training Loop ex:
#Epoch 1, Training loss 80.5940, Validation loss 271.2241
#Epoch 1000, Training loss 3.4822, Validation loss 13.0070
#Epoch 2000, Training loss 2.4849, Validation loss 7.2142
#Epoch 3000, Training loss 2.4375, Validation loss 6.2846

    print('\n')
    print(tloop)    # prints None, isn't this model?
    print(linear_model.weight)
    print(linear_model.bias)    
#None
#Parameter containing:
#tensor([[5.0868]], requires_grad=True)
#Parameter containing:
#tensor([-15.7719], requires_grad=True)

    print('\nEx2 End')



#########################
#########################
# NN Example
def ex3():
    print('\nEx3')

    # 4 layer/module NN
    seq_model = nn.Sequential(
                nn.Linear(1, 13), # output and next layer input MUST MATCH
                nn.Tanh(),
                nn.Linear(13, 1)) # linear comb/reg. layer
    print(seq_model)
#Sequential(
#  (0): Linear(in_features=1, out_features=13, bias=True)
#  (1): Tanh()
#  (2): Linear(in_features=13, out_features=1, bias=True)
#)

    # Inspect parameters w/ shape - Have to create list to inspect
    print([param.shape for param in seq_model.parameters()])
#[torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]

    # naming submodules with named_param. method
    for name, param in seq_model.named_parameters():
        print(name, param.shape)
#0.weight torch.Size([13, 1])
#0.bias torch.Size([13])
#2.weight torch.Size([1, 13])
#2.bias torch.Size([1])

    # Can also use OrderedDict library
    seq_model = nn.Sequential(OrderedDict([
        ('hidden_linear', nn.Linear(1, 8)),
        ('hidden_activation', nn.Tanh()),
        ('output_linear', nn.Linear(8,1))
    ]))
    print(seq_model)
#Sequential(
#  (hidden_linear): Linear(in_features=1, out_features=8, bias=True)
#  (hidden_activation): Tanh()
#  (output_linear): Linear(in_features=8, out_features=1, bias=True)
#)

    # Prev. gives more descriptive print
    for name, param in seq_model.named_parameters():
        print(name, param.shape)
#hidden_linear.weight torch.Size([8, 1])
#hidden_linear.bias torch.Size([8])
#output_linear.weight torch.Size([1, 8])
#output_linear.bias torch.Size([1])

#########################
#########################
# Monitoring Params
    # access params directly by using submodules as attributes
    print(seq_model.output_linear.bias)
#Parameter containing:
#tensor([-0.2567], requires_grad=True)

    # This is useful to monitor gradients during training
    optimizer = optim.SGD(seq_model.parameters(), lr=1e-3) # smaller lr for stability

    training_loop(
        n_epochs = 5000,
        optimizer = optimizer,
        model = seq_model,
        loss_fn = nn.MSELoss(),
        t_u_train = t_un_train,
        t_u_val = t_un_val,
        t_c_train = t_c_train,
        t_c_val = t_c_val)
#Epoch 1, Training loss 154.7812, Validation loss 440.8859
#Epoch 1000, Training loss 4.0427, Validation loss 50.9659
#Epoch 2000, Training loss 2.9794, Validation loss 30.9047
#Epoch 3000, Training loss 2.5858, Validation loss 21.9015
#Epoch 4000, Training loss 2.3535, Validation loss 16.3916
#Epoch 5000, Training loss 2.1948, Validation loss 12.4419


    print('output', seq_model(t_un_val))
    print('answer', t_c_val)        # check correct values at val. ind. 
    print('hidden', seq_model.hidden_linear.weight.grad) # check grad. for 1st layer
#output tensor([[23.4446],
#        [ 8.0311]], grad_fn=<AddmmBackward>)
#answer tensor([[28.],
#        [ 6.]])
#hidden tensor([[-0.0395],
#        [-0.0307],
#        [ 0.0137],
#        [ 0.0883],
#        [ 0.0115],
#        [-0.0088],
#        [ 0.0084],
#        [ 0.0093]])

    # Going to check gradients at every layer
    if(True):
        print('\nCHECKING GRADIENTS')
        # seq model has no weights in init
#        print('init:', seq_model.weight)
#        print('init:', seq_model.weight.grad)
        print('hidden:', seq_model.hidden_linear.weight) # weight tensor
        print('hidden:', seq_model.hidden_linear.weight.shape) # weight tensor
#hidden: torch.Size([8, 1])

        # act. fn has no weight attribute
#        print('act:', seq_model.hidden_activation.weight.grad)
        print('last:', seq_model.output_linear.weight)
        print('last:', seq_model.output_linear.weight.shape)
#last: torch.Size([1, 8])
        print('last grad:', seq_model.output_linear.weight.grad)
# Model output gave sizes already. Just wanted to check the gradients changing.
# Maybe there is a way to put this nicely into print epochs


#########################
#########################
# 6.3.3 Comparing to the linear model
    # plot comparison 
    t_range = torch.arange(20., 90.).unsqueeze(1)
    fig = plt.figure(dpi=600)
    plt.xlabel("Fahrenheit")
    plt.ylabel("Celsius")
    plt.plot(temp_u.numpy(), temp_c.numpy(), 'o')
    plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
    plt.plot(temp_u.numpy(), seq_model(0.1 * temp_u).detach().numpy(), 'kx')

    plt.savefig('Plot1')
    # NN is overfitting data

    print('\nEx3 End')


# Main
if __name__ == "__main__":
    # set seed
    torch.manual_seed(28)
    # set up dataset
    temp_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
    temp_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    temp_c = torch.tensor(temp_c).unsqueeze(1) # <1>
    temp_u = torch.tensor(temp_u).unsqueeze(1) # <1>

    n_samples = temp_u.shape[0]
    n_val = int(0.2 * n_samples)

    shuffled_indices = torch.randperm(n_samples)

    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]

#    print(train_indices, val_indices)
    t_u_train = temp_u[train_indices]
    t_c_train = temp_c[train_indices]

    t_u_val = temp_u[val_indices]
    t_c_val = temp_c[val_indices]

    t_un_train = 0.1 * t_u_train
    t_un_val = 0.1 * t_u_val

#    print(t_un_train, t_un_val)

######################################
######################################
# START CH. HERE 
    if(False):
        ex1()

# Linear Model function defaults to random weights and bias
# TEST CODE
    if(False):
        test()
        test2()

    if(False):
        ex2()

    if(True):
        ex3()

    print('\nMain End\n')
    


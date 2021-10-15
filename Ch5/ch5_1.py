import torch
from matplotlib import pyplot as plt 
import numpy as np
import torch.optim as optim

def model(t_u, w, b):
    """
    linear regression
    t_u: unknown values, w: param/weights tensor, b: bias tensor
    """
    return w * t_u + b


def loss_fn(t_obs, t_pred):
    squared_error = (t_obs - t_pred)**2
    return squared_error.mean() # MSE


def broadcasting_ex():
	x = torch.ones(())
	y = torch.ones(3, 1)
	z = torch.ones(1, 3)
	a = torch.ones(2, 1, 1)
	print(f"shapes: x: {x.shape}, y: {y.shape}")

	print(f"	z: {z.shape}, a: {a.shape}")
	print("x * y:", (x * y).shape)
	print("y * z:", (y * z).shape)
	print("y * z * a:", (y * z * a).shape)

	print('End of broadcasting ex')
	"""
shapes: x: torch.Size([]), y: torch.Size([3, 1])
        z: torch.Size([1, 3]), a: torch.Size([2, 1, 1])
x * y: torch.Size([3, 1])
y * z: torch.Size([3, 3])
y * z * a: torch.Size([2, 3, 3])
End of broadcasting ex

	"""


def dloss_fn(t_obs, t_pred):
# derivative of loss fn
    if(False):
        print(t_obs.size()) # torch.Size([11])
    dsq_error = 2 * (t_obs - t_pred) / t_obs.size(0) # div. is deriv. of mean
    return dsq_error


def dmodel_dw(obs, w, b):
# deriv. of model wrt w(weights) param.
    return obs

def dmodel_db(obs, w, b):
# deriv. of model wrt b(bias) param.
    return 1.0


def grad_fn(temp_u, temp_c, predicted, w, b):
    """
    Returns gradient of loss wrt to w and b
    """
    dloss_dpred = dloss_fn(predicted, temp_c)
    dloss_dw = dloss_dpred * dmodel_dw(temp_u, w, b)
    # chain rule
    dloss_db = dloss_dpred * dmodel_db(temp_u, w, b)
    # sum is the reverse of broadcasting we have to implicitly do 
    # when applying params. to entire vector of inputs, in model
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


def training_loop(n_epochs, learning_rate, params, temp_u, temp_c):
    """
    Update params for all training samples each epoch
    """
    for epoch in range(1, n_epochs + 1):
        w, b = params   # has to be vector of 2 elems 
        t_pred = model(temp_u, w, b)    # forward pass
        loss = loss_fn(t_pred, temp_c)
        grad = grad_fn(temp_u, temp_c, t_pred, w, b)    # backward pass
        # updates params based on gradient * learning rate
        params = params - learning_rate * grad  # like simple update ex.
        # print epoch info
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params


def training_loop_v(n_epochs, learning_rate, params, temp_u, temp_c,
                    print_params=True):
    """
    More verbose training loop
    """
    for epoch in range(1, n_epochs + 1):
        w, b = params

        t_pred = model(temp_u, w, b)
        loss = loss_fn(t_pred, temp_c)
        grad = grad_fn(temp_u, temp_c, t_pred, w, b)

        params = params - learning_rate * grad

        if epoch in {1, 2, 3, 10, 99, 100, 4000, 5000}:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            if print_params:
                print('     Params:', params)
                print('     Grad:  ', grad)
        if epoch in {4, 12, 101}:
            print('...')

        if not torch.isfinite(loss).all():
            break

    return params


##############################################
##############################################
# Ch. Pt. II - 5.5 Autograd helper function

def training_loop_auto1(n_epochs, learning_rate, params, temp_u, temp_c):
    """
    Temporary training loop with nograd
    """
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            # zero out gradient before differentiating
            params.grad.zero_()

        temp_p = model(temp_u, *params)
        loss = loss_fn(temp_p, temp_c)
        # backprop(chain rule) - find and store derivs.
        loss.backward()

        # for this ex.
        with torch.no_grad():
            # updating params tensor manually
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params


def training_loop_auto2(n_epochs, optimizer, params, temp_u, temp_c):
    """
	Training loop w/ optimizer module and autograd

# I think code this code is no longer needed since we zero out optim. now
		if params.grad is not None:
			params.grad.zero_()	
    """
    for epoch in range(1, n_epochs + 1):
        # set model and calc. loss
        t_pred = model(temp_u, *params) 
        loss = loss_fn(t_pred, temp_c)

        # zero out grad. before updating params
        optimizer.zero_grad()
        # backprop/chain 
        loss.backward()
        # update params
        optimizer.step()

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params


def training_loop_auto3(n_epochs, optimizer, params, train_t_u, val_t_u,
                        train_t_c, val_t_c):
    """
    Training loop w/ train and val/test set
    """ 
    for epoch in range(1, n_epochs + 1):
        # model w/ training set, then calc loss
        train_t_pred = model(train_t_u, *params)
        train_loss = loss_fn(train_t_pred, train_t_c)
        # check val model loss
        val_t_pred = model(val_t_u, *params)
        val_loss = loss_fn(val_t_pred, val_t_c) 

        # zero grad, backprop, then update params for model
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # print epoch
        if epoch <= 3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")
    return params

#        val_t_pred = model(val_t_u, *params)
#        val_loss = calc_forward()

def training_loop_final(n_epochs, optimizer, params, train_t_u, val_t_u,
						train_t_c, val_t_c):
    """
	Final training loop, w/ autograd, train and test sets, w/ no autograd for val set
    """
    for epoch in range(1, n_epochs + 1):
        # model training data and loss
        train_t_pred = model(train_t_u, *params)
        train_loss = loss_fn(train_t_pred, train_t_c)
        # Context manager - useful code
        # test validation data and loss w/ no autograd
        with torch.no_grad():
            val_t_pred = model(val_t_u, *params)
            val_loss = loss_fn(val_t_pred, val_t_c)
            assert val_loss.requires_grad == False  # forces no grad
        # zero out grad., backprop, update params.
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

		# print epoch info
        if epoch <= 3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")

    return params


def calc_forward(temp_u, temp_c, is_train):
    with torch.set_grad_enabled(is_train):
        t_pred = model(temp_u, *params)
        loss = loss_fn(t_pred, temp_c)
    return loss


def training_loop_short(n_epochs, optimizer, params, train_t_u, val_t_u,
                        train_t_c, val_t_c):
    """
    Short version w/ calc_forward - model/loss fn with requires_grad bool argument
    """
    for epoch in range(1, n_epochs + 1):
        # model training data and calc training loss using helper fn
        train_loss = calc_forward(train_t_u, train_t_c, True)
        # model val. data and calc val. loss
        val_loss = calc_forward(val_t_u, val_t_c, False)

        # zero out grad., backprop, update params
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()       
        # print epoch info
        if epoch <= 3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")
    return params


##############################################
##############################################
# Ch. Pt. II - 5.5 Autograd function Example fn Code
def auto_ex1():
    """
    Example code helper fn to run 1st ex. from 2nd half of book.
    For some reason, this fn keeps all vars from main, even though
    they aren't passed explicitely?
    """
    print('Start autograd example 1\n') 

    params = torch.tensor([1.0, 0.0], requires_grad=True)
    print(params.grad is None)
    print(params.grad)
    print(params)
# True
# None
# tensor([1., 0.], requires_grad=True)

    loss = loss_fn(model(temp_u, *params), temp_c)
    # differentiate
    loss.backward()
    
    print(params.grad)
# tensor([4517.2969,   82.6000])
    print(params)
# tensor([1., 0.], requires_grad=True)- stays same, gradients change


    print('\nRunning training loop auto1')
    # use temp training loop
    ex1 = training_loop_auto1(
        n_epochs = 5000,
        learning_rate = 1e-2,
        params = torch.tensor([1.0, 0.0], requires_grad=True),
        temp_u = temp_un,
        temp_c = temp_c
    )
    print(ex1)

#Running training loop auto1
#Epoch 500, Loss 7.860115
#Epoch 1000, Loss 3.828538
#Epoch 1500, Loss 3.092191
#Epoch 2000, Loss 2.957698
#Epoch 2500, Loss 2.933134
#Epoch 3000, Loss 2.928648
#Epoch 3500, Loss 2.927830
#Epoch 4000, Loss 2.927679
#Epoch 4500, Loss 2.927652
#Epoch 5000, Loss 2.927647
#tensor([  5.3671, -17.3012], requires_grad=True)

    print('\nEnd autograd example 1\n') 


def auto_ex2():
    """
    2nd ex for 2nd half of ch.
    """
    DEBUG1 = False 

    print('\nStart autograd example 2\n') 
    # check optimizers
    print(dir(optim))
#['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 
#'SGD', 'SparseAdam', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', 
#'__package__', '__path__', '__spec__', '_multi_tensor', 'functional', 'lr_scheduler', 'swa_utils']

# Just realized idk how this fn has access to the main vars without
# directly passing it. Maybe since I called this fn in main, Python remembers
# the variables. Not sure if this is good code. That's why I wanted
# to keep it all in main, for now. 
    if(DEBUG1):
        print('\n\nTest')
        print(temp_u)
        print(temp_un)

    params = torch.tensor([1.0, 0.0], requires_grad=True)
    learning_rate = 1e-5
    optimizer = optim.SGD([params], lr=learning_rate)
# Needs params as list OR gives error
#TypeError: params argument given to the optimizer should be an iterable of Tensors or dicts,
# but got torch.FloatTensor
    print(optimizer)
#SGD (
#Parameter Group 0
#    dampening: 0
#    lr: 1e-05
#    momentum: 0
#    nesterov: False
#    weight_decay: 0
#)

    t_pred = model(temp_u, *params)
    loss = loss_fn(t_pred, temp_c)
    loss.backward() # backprop/chain rule

# missing zeroing out gradient, or else grads. will accumulate - just for ex.

    optimizer.step()    # pytorch updates params 
    print(params)
#tensor([ 9.5483e-01, -8.2600e-04], requires_grad=True)

    # test training loop code
    # HYPER PARAMS
    # set params, learning rate, and optimizer
    print('\nTest Training loop')
    params = torch.tensor([1.0, 0.0], requires_grad=True)
    learning_rate = 1e-2
    optimizer = optim.SGD([params], lr=learning_rate)
    # set model on normalized data and find loss
    t_pred = model(temp_un, *params)
    loss = loss_fn(t_pred, temp_c)
    # set optimzer(zero grad. first!), then update params
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print params tensor for debug
    print(params)
#Test Training loop
#tensor([1.7761, 0.1064], requires_grad=True)


    # training loop auto 2 - w/ autograd and optim
    print('\nRunning Training Loop 2 - autograd and optim')

    params = torch.tensor([1.0, 0.0], requires_grad=True)
    learning_rate = 1e-2
    optimizer = optim.SGD([params], lr=learning_rate)

    training_loop_auto2(
        n_epochs=5000,
        optimizer=optimizer,
        params=params,
        temp_u=temp_un,
        temp_c=temp_c
    )
    print(params)
#Epoch 500, Loss 7.860120
#Epoch 1000, Loss 3.828538
#Epoch 1500, Loss 3.092191
#Epoch 2000, Loss 2.957698
#Epoch 2500, Loss 2.933134
#Epoch 3000, Loss 2.928648
#Epoch 3500, Loss 2.927830
#Epoch 4000, Loss 2.927679
#Epoch 4500, Loss 2.927652
#Epoch 5000, Loss 2.927647
#tensor([  5.3671, -17.3012], requires_grad=True)
# SAME result as autograd_ex1, where we computed grad. manually


    # testing out other optimizer
    print('\nTrying out optimizer=Adam')
    params = torch.tensor([1.0,0.0], requires_grad=True)
    learning_rate = 1e-1    # don't need small learning rate for Adam
    optimizer = optim.Adam([params], lr=learning_rate)

    training_loop_auto2(
        n_epochs=2000,
        optimizer=optimizer,
        params=params,
        temp_u=temp_u,  # don't need normalized data
        temp_c=temp_c
    )
    print(params)
#Trying out optimizer=Adam
#Epoch 500, Loss 7.612900
#Epoch 1000, Loss 3.086700
#Epoch 1500, Loss 2.928579
#Epoch 2000, Loss 2.927644
#tensor([  0.5367, -17.3021], requires_grad=True)

    print('\nEnd autograd example 2\n') 


def auto_ex3():
    print('\nStart autgrad example 3\n')

    DEBUG1 = False 
    DEBUG2 = True

    # Split dataset into train and validation/test set
    n_samples = temp_u.shape[0]
    n_val = int(0.2 * n_samples)
    if(DEBUG1):
        print(temp_u.shape, n_samples, n_val)   # torch.Size([11]) 11 2

    shuffled_indices = torch.randperm(n_samples)    # randomize indices
    if(DEBUG1):
        print(shuffled_indices, shuffled_indices.dtype) 
# tensor([ 5,  6,  7,  4,  2,  1,  3,  0,  8,  9, 10]) torch.int64

    # set splits based on randomly shuffled indices
    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]
    print(train_indices, val_indices)
# tensor([ 1,  8,  0,  6,  3, 10,  2,  4,  5]) tensor([9, 7])


    # create sets
    train_t_u = temp_u[train_indices]
    train_t_c = temp_c[train_indices]

    val_t_u = temp_u[val_indices]
    val_t_c = temp_c[val_indices]
    if(DEBUG2):
        print('\n')
        print(train_t_u, train_t_c)
        print('\n')
        print(val_t_u, val_t_c)

    # naive normalize
    train_t_un = 0.1 * train_t_u
    val_t_un = 0.1 * val_t_u


    # run training loop 3 on train and val sets
    print('\nRunning training loop 3 on train and val sets')
    params = torch.tensor([1.0, 0.0], requires_grad=True)
    learning_rate = 1e-2
    optimizer = optim.SGD([params],lr=learning_rate)

    training_loop_auto3(
        n_epochs = 3000,
        optimizer = optimizer,
        params = params,
        train_t_u = train_t_un,
        val_t_u = val_t_un,
        train_t_c = train_t_c,
        val_t_c = val_t_c
    )
    print(params)
#Running training loop 3 on train and val sets
#Epoch 1, Training loss 97.1315, Validation loss 4.9121
#Epoch 2, Training loss 39.5736, Validation loss 7.6988
#Epoch 3, Training loss 32.7771, Validation loss 15.8679
#Epoch 500, Training loss 8.4764, Validation loss 2.6401
#Epoch 1000, Training loss 3.9649, Validation loss 1.4434
#Epoch 1500, Training loss 3.1063, Validation loss 2.2897
#Epoch 2000, Training loss 2.9429, Validation loss 2.9192
#Epoch 2500, Training loss 2.9118, Validation loss 3.2435
#Epoch 3000, Training loss 2.9059, Validation loss 3.3943
#tensor([  5.4996, -18.1371], requires_grad=True)

    
    # run final training loop on train and val sets
    print('\nRunning final training loop on train and val sets')
    params = torch.tensor([1.0, 0.0], requires_grad=True)
    learning_rate = 1e-2
    optimizer = optim.SGD([params],lr=learning_rate)

    training_loop_final(
        n_epochs = 3000,
        optimizer = optimizer,
        params = params,
        train_t_u = train_t_un,
        val_t_u = val_t_un,
        train_t_c = train_t_c,
        val_t_c = val_t_c
    )
    print(params)
# works

    # run final training loop on train and val sets
    print('\nRunning short training loop on train and val sets')
    params = torch.tensor([1.0, 0.0], requires_grad=True)
    learning_rate = 1e-2
    optimizer = optim.SGD([params],lr=learning_rate)

# NOT WORKING - should have same output as prev. 2 fns
#    training_loop_short(
#        n_epochs = 3000,
#        optimizer = optimizer,
#        params = params,
#        train_t_u = train_t_un,
#        val_t_u = val_t_un,
#        train_t_c = train_t_c,
#        val_t_c = val_t_c
#    )
#    print(params)

    print('\nEnd autgrad example 3\n')


##############################################
##############################################
# Main
if __name__ == '__main__':
    DEBUG1 = False 
    B_EX = False

    # data
    # temp in celsius
    temp_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
    # temp in unknon units
    temp_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    # turn to tensor
    temp_c = torch.tensor(temp_c)
    temp_u = torch.tensor(temp_u)

#   idk?
#    plt.plot(temp_c.numpy(), temp_u.numpy())

#    test = model(temp_u, 0, 0)
#    print(test)

    # Def. ctor
    w = torch.ones(())  # tensor(1.)
    b = torch.zeros(()) # tensor(0.)

# just testing the diff. between this an w    
    t = torch.ones(1)


    print(w) # tensor(1.)
    print('\n')
    print(b) # tensor(0.)

    if(DEBUG1):
        print(t)        # tensor([1.])
        print(w.shape)  #    torch.Size([])
        print(t.shape)  #    torch.Size([1])
        temp = torch.ones((()))
        print(temp, temp.shape) # tensor(1.) torch.Size([])

    # basic test model
    t_pred = model(temp_u,w, b) 
    print(t_pred)
#tensor([35.7000, 55.9000, 58.2000, 81.9000, 56.3000, 48.9000, 33.9000, 21.8000,
#        48.4000, 60.4000, 68.4000])

    loss = loss_fn(t_pred, temp_c)
    print(loss) # tensor(1763.8848)


##############################################
##############################################
# Broadcasting example
    if(B_EX):   
        print('\n\n\n')
        broadcasting_ex()

    
##############################################
##############################################
    # 5.4.1 - basic parameter-update step
    print('\n5.4.1 ex:')
    # estimating rate of change of loss,wrt each param by subtracting small #
    delta = 0.1

    loss_rate_of_change_w = \
        (loss_fn(model(temp_u, w + delta, b), temp_c) - 
         loss_fn(model(temp_u, w - delta, b), temp_c)) / (2.0 * delta)
    print(loss_rate_of_change_w)    # tensor(4517.2974)

    # introduce how much params. change when updated w/ learning rate
    learning_rate = 1e-2
    # single param lin. reg.
    w = w - learning_rate * loss_rate_of_change_w 
    # compute rate for b again 
    loss_rate_of_change_b = \
        (loss_fn(model(temp_u, w + delta, b), temp_c) - 
         loss_fn(model(temp_u, w - delta, b), temp_c)) / (2.0 * delta)
    print(loss_rate_of_change_b)    # tensor(-261170.)
    b = b - learning_rate * loss_rate_of_change_b

    print(w) # tensor(-44.1730)
    print(b) # tensor(2611.7000)

##############################################
##############################################
    # 5.4.2
    print('\n5.4.2 ex:')

    # test new fn
    print(dloss_fn(t_pred, temp_c)) 
# tensor([6.4000, 7.6182, 7.8545, 9.8000, 8.2364, 7.4364, 5.6182, 4.6909, 7.7091,
#            8.6182, 8.6182])

    # test gradient fn, try to test this w/ test vectors
    print(grad_fn(temp_u, temp_c, t_pred, w, b))
# tensor([4517.2964,   82.6000])

    if(False):
        temp_w = [3.0, 5.0, 7.0, 9.0]
        temp_b = [2.0, 4.0, 6.0, 8.0]
        temp_w = torch.tensor(temp_w)
        temp_b = torch.tensor(temp_b)

        print(grad_fn(temp_u, temp_c, t_pred, temp_w, temp_b))
        # tensor([4517.2964,   82.6000])
        # returns same since fn is nieve


##############################################
##############################################
# 5.4.3
    print('\n5.4.3 - training loop:')
    # overfitting ex.
    ex1 = training_loop(
        n_epochs = 100,
        learning_rate = 1e-2,
        params = torch.tensor([1.0, 0.0]),  # LSE might be good starting?
        temp_u = temp_u,
        temp_c = temp_c
    )
    print(ex1)  # tensor([nan, nan])
    # Very simple lin. reg., but learning rate overshoots, and model overfits until
    # inf. loss

    print('\n\n')
    # too small param. updates, loss stalls
    ex2 = training_loop_v(
        n_epochs = 100,
        learning_rate = 1e-4,
        params = torch.tensor([1.0, 0.0]),
        temp_u = temp_u,
        temp_c = temp_c
    )
    print(ex2)
#Epoch 98, Loss29.024492
#Epoch 99, Loss29.023582
#Epoch 100, Loss29.022667
# tensor([ 0.2327, -0.0438])

#Epoch 100, Loss 29.022667
#     Params: tensor([ 0.2327, -0.0438])
#     Grad:   tensor([-0.0532,  3.0226])
#tensor([ 0.2327, -0.0438])

    # change in loss stalls due to small learning rate


##############################################
##############################################
# 5.4.4 - Normalizing inputs

    # naive normalizing to make input close to [-1,1]
    # This helps each param. gradient stay on a similar scale
    print('\n5.4.4 - training loop:')
    temp_un = 0.1 * temp_u
    print(temp_un, temp_un.dtype)
#tensor([3.5700, 5.5900, 5.8200, 8.1900, 5.6300, 4.8900, 3.3900, 2.1800, 4.8400,
#        6.0400, 6.8400]) torch.float32
    print(temp_un.shape)
#torch.Size([11])
    
    ex3 = training_loop_v(
        n_epochs = 100,
        learning_rate = 1e-2,
        params = torch.tensor([1.0, 0.0]),
        temp_u = temp_un,
        temp_c = temp_c
    )
    print(ex3)
#Epoch 100, Loss 22.148710
#     Params: tensor([ 2.7553, -2.5162])
#     Grad:   tensor([-0.4446,  2.5165])
#tensor([ 2.7553, -2.5162])

    # We can see that after a naive normalization, the loss/error is lower
    # and the gradients are closer in scale

    # change epochs to 5000
    params  = training_loop_v(
        n_epochs = 5000,
        learning_rate = 1e-2,
        params = torch.tensor([1.0, 0.0]),
        temp_u = temp_un,
        temp_c = temp_c
    )
#Epoch 5000, Loss 2.927648
#     Params: tensor([  5.3671, -17.3012])
#     Grad:   tensor([-0.0001,  0.0006])
    print('Params: ',params)
    print('\n\n ')

##############################################
##############################################
# 5.4.5 - Visualizing
 
    print('\n5.4.5 = Visualizing ')
    # temp. pred. model 
    t_pred = model(temp_un, *params) 

    # using matplotlib
    fig = plt.figure(dpi=600)
    plt.xlabel("Temperature (*Fahrenheit)")
    plt.ylabel("Temperature (*Celsius)")
    plt.plot(temp_u.numpy(), t_pred.detach().numpy())
    plt.plot(temp_u.numpy(), temp_c.numpy(), 'o')

    # saves plot image in folder
    plt.savefig('Plot1')


##############################################
##############################################
# 5.5 Autograd

    # run auto example function
    print('\n\nRunning autograd example 1\n')
    auto_ex1()
    
    print('\n\nRunning autograd example 2\n')
    auto_ex2()


    print('\n\nRunning autograd example 3\n')
    auto_ex3()





    print('End of main\n\n')



##############################################
##############################################
    '''
comment section:
Notes:

I should have better code cleanliness. I should put debugs everywhere but try to
keep it neat. Also, I should really set up my margins so I understand how to format
my work. Coding is fun. Just need to practice and try my best. 

I like the music.

Also, 

1:40 AM:
    Here.
    Be a hacker like George HOtz. lol

1:49 AM:
    How do I search in Vim. Damn, maybe I should stay on my app since Vim doesn't
    have autocorrect.

2:29 Feb 14,2020:
    Message to myself: Sorry for the meh code orginization. I've been neat w/ funs
    but the book had some giant ex. I split it into funs with autograd, 2nd half
    of chapter. 
    '''




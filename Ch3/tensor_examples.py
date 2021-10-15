# tensor_examples.py
# PyTorch 1.7.0-CPU Anacondo3-2020.02 Python 3.8.3
# Ubuntu
# Taken from https://visualstudiomagazine.com/articles/2020/06/09/working-with-pytorch.aspx

import numpy as np
import torch as T
device = T.device("cuda")

# ---------------------------------------------------------

DEBUG1 = True
DEBUG2 = True
DEBUG3 = True
DEBUG4 = True

def test1():
    print("\nBegin PyTorch tensor examples ")

    np.random.seed(1)
    T.manual_seed(1)
    T.set_printoptions(precision = 2)


# 1. Creating from numpy or list

    a1 = np.array([[0, 1, 2],
                 [3, 4, 7]], dtype = np.float32)
    t1 = T.tensor(a1, dtype = T.float32).to(device)
    t2 = T.tensor([0, 1, 2, 3],
    dtype = T.int64).to(device)
    t3 = T.zeros((3, 4), dtype = T.float32)

    x = t1[1][0]
    v = x.item()
    if(DEBUG1):
      print(x)      # tensor(3., device='cuda:0')
      print(v)      # 3.0 
      
#  print(t3)
#  print(t3.size())

#print tensors
    if(DEBUG1):
      print(a1)     # [[0. 1. 2.]
                    #  [3. 4. 7.]]
      print('\n\n')  
      print(t1)     # tensor([[0., 1., 2.],
                    #         [3., 4., 7.]], device='cuda:0')
      print('\n')
      print(t2)     # tensor([0, 1, 2, 3], device='cuda:0')


# 2. shape, reshape, view, flatten, squeeze 
    
    print(t1.shape)             # torch.Size([2, 3]) 
    t4 = t1.reshape(1, 3, 2)    # or t1.view(1, 3, 2)        z x m x n matrix
    t5 = t1.flatten()           # or T.flatten(t1)           turns into vector
    t6 = t4.squeeze()           # or T.reshape(t4, (3, -1))  
                                # removes any dim. of size 1, cols. only I think(lin. alg) 

    if(DEBUG2):
        print('\n\n\n #2')
        print(t4)               # tensor([[[0., 1.],
                                #          [2., 3.],
                                #          [4., 7.]]], device='cuda:0')
        print(t5)               # tensor([0., 1., 2., 3., 4., 7.], device='cuda:0')
        print(t6)               # tensor([[0., 1.],
                                #         [2., 3.],
                                #         [4., 7.]], device='cuda:0')


# 3. tensor to numpy or list

    temp = T.tensor(t1, dtype = T.float32).to('cpu')
#    print(temp)
    a2 = temp.numpy()           # t1.detach().numpy()
    lst1 = t1.tolist()
    if(DEBUG3):
        print('\n\n\n #3')
        print(a2)
        print(lst1)             # [[0.0, 1.0, 2.0], [3.0, 4.0, 7.0]]

# 4. functions

    print('\n\n')
    t7 = T.add(t1, 2.5)         # t1 = t1 + 2.5
    if(DEBUG4):
        print(t7)               # tensor([[2.50, 3.50, 4.50],
                                #         [5.50, 6.50, 9.50]], device='cuda:0')
    t7.add_(3.5)
    if(DEBUG4):
        print(t7)               # tensor([[ 6.,  7.,  8.],
                                #         [ 9., 10., 13.]], device='cuda:0')

    (big_vals, big_idxs) = T.max(t1, dim = 1)
    print(big_vals)             # tensor([2., 7.], device='cuda:0')
                                # gives max column
    probs = T.softmax(t1, dim = 1)
    print(probs[0])             # tensor([0.09, 0.24, 0.67], device='cuda:0')
    if(DEBUG4):
        print(probs)            # tensor([[0.09, 0.24, 0.67],
                                #         [0.02, 0.05, 0.94]], device='cuda:0')

    print('\n End demo')
# ---------------------------------------------------------

if __name__ == "__main__":
    test1()


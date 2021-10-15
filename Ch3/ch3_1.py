import torch
# import random # doesn't work for tensors?

#tensor practice

def ex1():
    a = torch.ones(3)
# a.item() # only works for single item tensors
    print(a)
    print(a.size())
    print(a[1])     #tensor(1.) floating point

    flo = float(a[1])
    print(flo)      # 1.0

# tensor is mutable
    a[2] = 2.0
    print(a)        # tensor([1., 1., 2.])

    a[2] = 2
    print(a)        # tensor([1., 1., 2.])

    points = torch.zeros(6)
    for i in range(6):
        points[i] = float(i)

    print(points)       # tensor([0., 1., 2., 3., 4., 5.])
    print(a.dtype)      # torch.float32

# get coordinates of first point
    print(float(points[0]), float(points[1]))       # 0.0 1.0

# make 2D Tensor
    points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    print(points)       # tensor([[4., 1.],
                        #  		  [5., 3.],
                        #		  [2., 1.]])

    print(points.shape)     # torch.Size([3, 2]) 1 x 3 x 2 matrix
    print(points[1, 1])     # tensor(3.)
    print(points[0])        # tensor([4., 1.])

def ex2():
# pytorch can use range indexing notation like Python lists
    points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    print(points[1:])       # all rows after first, implicitely all cols.
    print(points[1:, :])    # all rows after first, all cols.
    print(points[1:, 0])    # all rows after first, 0th col. as 1D tensor
    print(points[None])     # adds dim. of size 1, like unsqueeze
                            # tensor([[[4., 1.],
        					#  		   [5., 3.],
        					#		   [2., 1.]]])
    # doesn't work
#    print((points[None].size()))
    none = points[None]
    print(none.size())      # torch.Size([1, 3, 2]) 1 x 3 x 2, 3D tensor
    print(points.size())    # torch.Size([3, 2])

# print shape of tensor
    print('test \n')
    print(points.shape)     # torch.Size([3, 2])
    print(none.shape)       # torch.Size([1, 3, 2])


def ex3():
    # set seed
    torch.manual_seed(0)

    # image tensor ex.
    img_t = torch.randn(3, 5, 5)    # shape [channels, rows, columns] 
    weights = torch.tensor([0.2126, 0.7152, 0.0722])    # typical weights for colors to derive
                                                        # single brightness value
    print(weights.size())       # torch.Size([3])
    print(weights.shape)        # torch.Size([3])
    none = weights[None]
    print(none.shape)           # torch.Size([1, 3])

    # these are 1D tensors. Not a 1 x n scalar matrix.
    test = torch.tensor([1, 2])
    print(test.shape)           # torch.Size([2])

    # ex. of 2 batch of images
    batch_t = torch.randn(2, 3, 5, 5)   # shape [batch, channels, rows, col.]

    # lazy unweighted mean
    print('\nPrinting img means:')
    # based on finding mean of [0], or RGB channel
    img_gray_naive = img_t.mean(-3)
    batch_gray_naive = batch_t.mean(-3)
    print(img_gray_naive.shape, batch_gray_naive.shape) # torch.Size([5, 5]) torch.Size([2, 5, 5])
                                                        # lost RGB dim. since it's now avgd. out

    # next ex.
    print('\n Tensor mult:')
    unsqueezed_weights = weights.unsqueeze(-1).unsqueeze(-1)    # adds two dims of 1 to end of tensor dim.
    img_weights = (img_t * unsqueezed_weights)  # [3, 5, 5] * [3, 1, 1] 
    batch_weights = (batch_t * unsqueezed_weights)  # [2, 3, 5, 5] * [1, 3, 1, 1]i
                                            # BROADCASTING: Appends leading dim. of size 1 when mult.
    img_gray_weighted = img_weights.sum(-3)
    batch_gray_weighted = batch_weights.sum(-3)
    print(batch_weights.shape, batch_t.shape, unsqueezed_weights.shape)
    #  torch.Size([2, 3, 5, 5]) torch.Size([2, 3, 5, 5]) torch.Size([3, 1, 1])

    # DEBUG
    DEBUG1 = False
    if(DEBUG1):
        print('\n\n')
        print(unsqueezed_weights)
#tensor([[[0.2126]],
#        [[0.7152]],
#        [[0.0722]]])
        print(unsqueezed_weights.size())     # 3 x 1 x 1, 3D tensor
        print('\n')
        print(img_t)
        print(img_t.shape)      # torch.Size([3, 5, 5]), 3D tensor
        print(img_weights)

    # named tensors - experimental feature
    named_weights = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
    print(named_weights)    # tensor([0.2126, 0.7152, 0.0722], names=('channels',))

    # refine names method - better than prev., works on already made tensors
    # ... allows leaving out dim. 
    img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
    batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')
    print("img named:", img_named.shape, img_named.names)  
    #  img named: torch.Size([3, 5, 5]) ('channels', 'rows', 'columns')
    print("batch named:", batch_named.shape, batch_named.names)
    #  batch named: torch.Size([2, 3, 5, 5]) (None, 'channels', 'rows', 'columns')

    # Method to align missing dim., changes tensor to match right order
    weights_aligned = named_weights.align_as(img_named)  # [3] -> [3, 1, 1]
    print(weights_aligned.shape, weights_aligned.names)
#    torch.Size([3, 1, 1]) ('channels', 'rows', 'columns')
    print(weights_aligned)
# tensor([[[0.2126]],
#         [[0.7152]],
#         [[0.0722]]], names=('channels', 'rows', 'columns'))

    # ex. cont.-sum method can take named dim.
    gray_named = (img_named * weights_aligned).sum('channels')  # [3, 5, 5] * [3, 1, 1]
    print(gray_named.shape, gray_named.names)	# torch.Size([5, 5]) ('rows', 'columns')

# gives error - can't sum dims. w/ diff. names
# gray_named = (img_named[..., :3] * weights_named).sum('channels')




    # turn back to unnamed tensors
    gray_plain = gray_named.rename(None)
    print(gray_plain.shape, gray_plain.names)   # torch.Size([5, 5]) (None, None)


def ex4():
    # default type is float32 and int64, can set dtype as arg. 
    double_points = torch.ones(10, 2, dtype=torch.double)
    short_points = torch.tensor([[1, 2],[3, 4]], dtype=torch.short)
    print(double_points.dtype)  # torch.float64
    print(short_points.dtype)   # torch.int16

    # changing types - use torch.type for easy typing
    double_points = torch.zeros(10, 2).to(torch.double)
    short_points = torch.zeros(10, 2).to(dtype=torch.short)

    # for operations mixing types, torch auto converts to larger type
    points_64 = torch.rand(5, dtype=torch.double)
    points_short = points_64.to(torch.short)
    mult = points_64 * points_short
    print(mult.dtype)   # torch.float64
    print(mult) # tensor([0., 0., 0., 0., 0.], dtype=torch.float64)


def ex5():
    a = torch.ones(3, 2) 
    a_t = torch.transpose(a, 0, 1)

    print(a_t, a_t.shape)
# tensor([[1., 1., 1.],
#         [1., 1., 1.]]) torch.Size([2, 3])

	# OR
    a_t = a.transpose(0, 1)
    print(a_t)

    print(a.storage())
# 1.0
# 1.0
# 1.0
# 1.0
# 1.0
# 1.0
# [torch.FloatStorage of size 6]

    # tensor storage is mutable too
    points = torch.tensor([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]) 
    a_storage = points.storage()
    a_storage[0] = 2.0  # changes t
    print(points)  
    print(a_storage.dtype) # torch.float32

    # methods of Tensor objects
    points.zero_()  # turns into zero tensor
    print(points)

    # storage properties: storage_offset(), size(), stride()
    print(points.stride())  # (2, 1)

    
    # Transposing
    some_t = torch.ones(3, 4, 5)
    transpose_t = some_t.transpose(0, 2)
    print(some_t.shape, transpose_t.shape)  # torch.Size([3, 4, 5]) torch.Size([5, 4, 3])
    print(some_t.stride(), transpose_t.stride())    #  (20, 5, 1) (1, 5, 20)
											# def. of transpose is flipping stride in storage    
    print(some_t)
    print('\n')
    print(transpose_t)
'''
tensor([[[1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.]],

        [[1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.]],

        [[1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.]]])


tensor([[[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]]])
'''


def ex6():
    # specifying torch dtype in argument
    double_points = torch.ones(10, 2, dtype=torch.double)
    short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short) # int tensor

    print(double_points.dtype)  # torch.float64
    print(short_points.dtype)   # torch.int16
    
#    print(double_points)
    
    # to tensor method for type casting
    double_points = torch.zeros(10, 2).to(torch.double)
    short_points = torch.zeros(10,2).to(dtype=torch.short) 
    print(double_points.dtype, short_points.dtype)  # torch.float64 torch.int16
   
    # tensor type mixing coverts to largest type
    points_64 = torch.rand(5, dtype=torch.double)
    points_short = points_64.to(torch.short)
    print(points_64 * points_short) # tensor([0., 0., 0., 0., 0.], dtype=torch.float64)

    # Tensor api - transposing
    a = torch.ones(3, 2)
    # transpose a by 0 and 1 cols.
    a_t = torch.transpose(a, 0, 1)
    print(a.shape, a_t.shape)   # torch.Size([3, 2]) torch.Size([2, 3])
    # also by
    a_t = a.transpose(0, 1)
    print(a_t.shape)

    print('\n\n Transpose ex: \n')

    # More transposing
    points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    print(points)
#tensor([[4., 1.],
#        [5., 3.],
#        [2., 1.]])

	# transpose short hand w/ NO arguments
    points_t = points.t()
    print(points_t)
#tensor([[4., 5., 2.],
#        [1., 3., 1.]])

	# verify transpose operation works on same storage
    print(id(points.storage()) == id(points_t.storage()))   # True
    print(points.stride(), points_t.stride())   # (2, 1) (1, 2)

    # Transpose in higher dims.
    some_t = torch.ones(3, 4, 5)
    transpose_t = some_t.transpose(0, 2)
    print(some_t.shape, transpose_t.shape)  # torch.Size([3, 4, 5]) torch.Size([5, 4, 3])

    # def. of transpose is changing row/col. stride in storage
    print(some_t.stride(), transpose_t.stride()) # (20, 5, 1) (1, 5, 20)

    print(some_t)
    print('\n')

# tensor([[[1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.]],
  
#       [[1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.]],

#        [[1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.]]])

    print(transpose_t)

# tensor([[[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]],

#        [[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]],

#        [[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]],

#        [[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]],

#        [[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]]])

    # NumPy conversion - very efficient w/ PyTorch
    points = torch.ones(3, 4)
    points_np = points.numpy()
    print(points_np, points_np.dtype)
#[[1. 1. 1. 1.]
# [1. 1. 1. 1.]
# [1. 1. 1. 1.]] float32


	# see book for putting tensors in GPU/ CPU

def ex7():
   a = torch.ones(2, 4, 3, 3) 
   print(a)

'''
tensor([[[[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]]],


        [[[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]]]])
'''


if __name__ == "__main__":
#    random.seed(28)    # doesn't work for tensors
    ex7()
    print("got here")






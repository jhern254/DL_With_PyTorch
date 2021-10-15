import torch
import imageio
import os
import csv
import numpy as np

def set_path(directory):
    DEBUG = True
    #set wd
    path = ('/home/jun/Documents/Programming/DL_With_PyTorch/Book_git/dlwpt-code/data/p1ch4/'
             + directory + '/')
    os.chdir(path)
    if(DEBUG):
        print(os.getcwd())


def ex1():
    DEBUG = False

    set_path('image-dog')
    img_arr = imageio.imread('bobby.jpg')
    print(img_arr.shape)    # (720, 1280, 3),(W,H,Ch) NumPy like array
    print(img_arr.dtype)    # uint8

    img = torch.from_numpy(img_arr)     # changes storage of img
    out = img.permute(2, 0, 1)          # PyTorch needs (Ch, H, W)     

    # set up pre-allocated tensor for image batch
    batch_size = 3
    batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8) 
    # 3 RGB images, 256 x 256. Most standard images are 8-bit.

    set_path('')
    data_dir = 'image-cats/' 
    filenames = [name for name in os.listdir(data_dir)
                    if os.path.splitext(name)[-1] == '.png']

    print(filenames) # ['cat2.png', 'cat3.png', 'cat1.png']


    for i, filename in enumerate(filenames):
        img_arr = imageio.imread(os.path.join(data_dir, filename)) 
#        print(len(img_arr)) # 3 images
        img_t = torch.from_numpy(img_arr)   # img_arr is NumPy obj 
        img_t = img_t.permute(2, 0, 1)      # fix order (Ch, H, W)
        img_t = img_t[:3]                   # keep only first 3 channels
        batch[i] = img_t 
        if(DEBUG):
            print(img_arr)
            print(filename, i)  # 3 RGB files
            print('\n\n')

    print(img_arr.dtype)    # uint8. Normal NumPy array 
    print(batch.dtype)      # torch.uint8. Should be in float for NN 
    
    # turn tensor to float, then normalize based on pixels for 8-bit unsigned
    norm = batch.clone()
    norm = norm.float()       # .to(torch.float)
    print(norm.dtype)          # torch.float32
    norm /= 255.0

    if(DEBUG):
        print(batch[1])
        print('\n\n')
        print(norm[1])

    # OR
    # can standardize based on std. Normal
    stdn = batch.clone()
    stdn = stdn.float()
    n_channels = stdn.shape[1]
    if(DEBUG):
        print(n_channels) # 3

    # standardize every dim. 
    for c in range(n_channels):
        mean = torch.mean(stdn[:, c])
        std = torch.std(stdn[:, c])
        stdn[:, c] = (stdn[:, c] - mean) / std

    print('\n\nComparing normalized vs standardized\n')
    print(norm[1], norm.dtype)
    print('\n\n')
    print(stdn[1], stdn.dtype)

    print('Got Here')


def ex2():
    # volumetric data - CT Scans for project
    set_path('volumetric-dicom/')
    dir_path = "2-LUNG 3.0  B70f-04083"
    vol_arr = imageio.volread(dir_path, 'DICOM')
    print(vol_arr.shape)    # (99, 512, 512)

    # 5D tensor for vol. data - (Batch, depth, ch(usually 1 since gray), H, W)
    # need to reformat to (Ch, D, H, W)
    vol = torch.from_numpy(vol_arr).float()
    vol = vol.unsqueeze(0)

    print(vol.shape)    # torch.Size([1, 99, 512, 512])
    # only 1 image

    print(len(vol_arr)) # 99. 99 Depth?

    print(vol[:1])

    print('End ex2')


def ex3():
    # tabular data ex.

    set_path('tabular-wine')
    wine_path = "winequality-white.csv"
    wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";",
                            skiprows=1)

    print(wineq_numpy, wineq_numpy.dtype)
#[[ 7.    0.27  0.36 ...  0.45  8.8   6.  ]
# [ 6.3   0.3   0.34 ...  0.49  9.5   6.  ]
# [ 8.1   0.28  0.4  ...  0.44 10.1   6.  ]
# ...
# [ 6.5   0.24  0.19 ...  0.46  9.4   6.  ]
# [ 5.5   0.29  0.3  ...  0.38 12.8   7.  ]
# [ 6.    0.21  0.38 ...  0.32 11.8   6.  ]] float32

    col_list = next(csv.reader(open(wine_path), delimiter=';'))

    print(wineq_numpy.shape, col_list)

#(4898, 12) ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
# 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
# 'sulphates', 'alcohol', 'quality']

# 4898 rows, 12 cols.

    # turn to torch tensor
    wineq = torch.from_numpy(wineq_numpy) 
    print(wineq.shape, wineq.dtype)      # torch.Size([4898, 12]) torch.float32

    # last col. is quality score. We will use as response var.(ground truth)
    # for classification
    data = wineq[:, :-1]    # select all rows, all cols. expcet last
    print(data, data.shape)

#tensor([[ 7.0000,  0.2700,  0.3600,  ...,  3.0000,  0.4500,  8.8000],
#        [ 6.3000,  0.3000,  0.3400,  ...,  3.3000,  0.4900,  9.5000],
#        [ 8.1000,  0.2800,  0.4000,  ...,  3.2600,  0.4400, 10.1000],
#        ...,
#        [ 6.5000,  0.2400,  0.1900,  ...,  2.9900,  0.4600,  9.4000],
#        [ 5.5000,  0.2900,  0.3000,  ...,  3.3400,  0.3800, 12.8000],
#        [ 6.0000,  0.2100,  0.3800,  ...,  3.2600,  0.3200, 11.8000]]) torch.Size([4898, 11])

    # set up response/target vector
    target = wineq[:, -1]   # select all rows, last col.
    print(target, target.shape) # tensor([6., 6., 6.,  ..., 6., 7., 6.]) torch.Size([4898])
    print(target.dtype)

    # transform target to int labels
    target = target.long()     # tensor([6, 6, 6,  ..., 6, 7, 6]) torch.int64
# book method
#    target = wineq[:, -1].long()    # tensor([6, 6, 6,  ..., 6, 7, 6]) torch.int64
    print(target, target.dtype)
#    print(target[1].shape)  # NA
    print(target.shape[0]) #4898 - accesses value in tensor shape
    print('\n\n')

    #################################################
    #################################################
    # One-hot encoding - IMPORTANT for classification. 
    # 1) Build zero tensor of same size
    classes = 10

    target_onehot = torch.zeros(target.shape[0], classes)   # 4898, 10 
    # scatter tensor method fn, fills tensor w/ values from another tensor 
    print(target_onehot.dtype)  # torch.float32
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
    # (dim for following args.-1, col. of elements to scatter, scalar(value) to scatter) 
    # unsqueeze is used since the tensor to copy needs to be same dim.
    print(target_onehot.shape, target_onehot.dtype)     # torch.Size([4898, 10]) torch.float32
    # this operation converted to float? No, def. was float32

    # print out tensors
    print(target_onehot[:10, :10])
    # compare tensors - they are same
    print(target[:10])
#tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])
#tensor([6, 6, 6, 6, 6, 6, 6, 6, 6, 6])

    # Unsqueeze mini example
    # have to have same dim. as target_onehot, when using scatter
#    print(target.shape) # torch.Size([4898])
    target_unsqueezed = target.unsqueeze(1)
#    print(target_unsqueezed.shape)  # torch.Size([4898, 1])


    #################################################
    #################################################
    print('\n\nSetting up data for modeling:\n')
    # orig. data tensor
    print(data, data.shape)
    data_mean = torch.mean(data, dim=0) # dim 0(rows) means
    data_var = torch.var(data, dim=0)
    # standardize data
    data_normalized = (data - data_mean) / torch.sqrt(data_var)
    print(data_normalized, data_normalized.shape)

    # find classification thresholds
    bad_indexes = target <= 3
    print(bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum())  
    # torch.Size([4898]) torch.bool tensor(20) - only 20 bad wines
    # subset bad data 
    bad_data = data[bad_indexes]
    print(bad_data.shape)   # torch.Size([20, 11])
    
    # split data into 3 categories
    bad_data = data[target <= 3]
    mid_data = data[(target > 3) & (target < 7)]
    good_data = data[target >= 7]

    bad_mean = torch.mean(bad_data, dim=0)
    mid_mean = torch.mean(mid_data, dim=0)
    good_mean = torch.mean(good_data, dim=0)

    # Print data w/ means. zip turns vars into tuple
    for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
        print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))
# 0 fixed acidity          7.60   6.89   6.73
# 1 volatile acidity       0.33   0.28   0.27
# 2 citric acid            0.34   0.34   0.33
# 3 residual sugar         6.39   6.71   5.26
# 4 chlorides              0.05   0.05   0.04
# 5 free sulfur dioxide   53.33  35.42  34.55
# 6 total sulfur dioxide 170.60 141.83 125.25
# 7 density                0.99   0.99   0.99
# 8 pH                     3.19   3.18   3.22
# 9 sulphates              0.47   0.49   0.50
#10 alcohol               10.34  10.26  11.42

    #################################################
    #################################################
    # EDA -  use total sulfur dioxide as crude criterion - this is good practice
    # crude prediction using cols.
    total_sulfur_threshold = 141.83 # avg of mid
    total_sulfur_data = data[:, 6]  # all rows, col. 6
    predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)  # less than
    print(predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())
    # torch.Size([4898]) torch.bool tensor(2727)

    # This threshold implies half wines are high quality
    # use crude est. as threshold of good wines
    actual_indexes = target > 5
    print(actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum())
    # torch.Size([4898]) torch.bool tensor(3258), tensor

    # testing if total sulfur is good predictor, and if picked threshold is good
    n_matches = torch.sum(actual_indexes & predicted_indexes).item() # Sum of values
    n_predicted = torch.sum(predicted_indexes).item()
    n_actual = torch.sum(actual_indexes).item()
    print(n_matches, n_predicted, n_actual) # 2018 2727 3258

    # nieve easy prediction model of seeing if total sulfur is good predictor
    print(n_matches, n_matches / n_predicted, n_matches / n_actual)
    # 2018 0.74000733406674 0.6193984039287906
    # prediction acc: 74%, valid acc: 61%
    print('End ex3')


def ex4():
    set_path('bike-sharing-dataset')

    # see book for data dict.
    bikes_numpy = np.loadtxt(
        "hour-fixed.csv",
        dtype=np.float32,
        delimiter=",",
        skiprows=1,
        converters={1: lambda x: float(x[8:10])})   
    # converts date strings to numbers(days) in col. 1
    bikes = torch.from_numpy(bikes_numpy)
    print(bikes.shape, bikes.dtype) # torch.Size([17520, 17]) torch.float32
    
    print(bikes.stride())   # (17, 1)
    print(bikes.is_contiguous())    # True
    # Use view to change tensor storage in batches of 24 hours
    # view arguments reshape data 
    daily_bikes = bikes.view(-1, 24, bikes.shape[1])
    print(daily_bikes.shape, daily_bikes.stride())  # torch.Size([730, 24, 17]) (408, 17, 1)
    # view changes how tensor looks at storage, thus changing stride based on hours
    # view(however indexes(rows) many left-used if unsure of row #, rows, cols)

    print(daily_bikes.is_contiguous())  # True 
    # transpose for N x C x L order for tensor
    daily_bikes = daily_bikes.transpose(2, 1)
    print(daily_bikes.shape, daily_bikes.stride()) # torch.Size([730, 17, 24]) (408, 1, 17)
    print(daily_bikes.is_contiguous())  # False, no longer contiguous

    #################################################
    #################################################
    # data prep.
    print('\n\nData Prep:\n')
    first_day = bikes[:24].long()
    print(first_day.shape)   # torch.Size([24, 17])
    # init. one hot tensor w/ zeros
    weather_onehot = torch.zeros(first_day.shape[0], 4)
    print(first_day[:, 9])  # print weather factor col.
# tensor([1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2])
    print(first_day[:, 9].shape)    # torch.Size([24])


	# One hot encoding
    print(weather_onehot.shape) # torch.Size([24, 4])
    weather_onehot.scatter_(
        dim=1,                                          # dim. for following args
        index=first_day[:,9].unsqueeze(1).long() - 1,   # -1 since weather var:1-4
        value=1.0)                                      # tensor value to fill
    # (dim for following args.-1, index of tensor to copy, scalar(value) to scatter) 
    # unsqueeze is used since the tensor to copy needs to be same dim.
    print(weather_onehot, weather_onehot.dtype) # torch.float32

    # Concatenate one hot tensor w/ orig. data tensor
    # just for first day, concat. along col. 1(appended)
    cat = torch.cat((bikes[:24], weather_onehot), 1)[:1]
    print(cat)
#tensor([[ 1.0000,  1.0000,  1.0000,  0.0000,  1.0000,  0.0000,  0.0000,  6.0000,
#          0.0000,  1.0000,  0.2400,  0.2879,  0.8100,  0.0000,  3.0000, 13.0000,
#         16.0000,  1.0000,  0.0000,  0.0000,  0.0000]])


    # one hot concat. ex. w/ daily bikes data tensor
    # have to start process over
    print(daily_bikes.shape, daily_bikes.is_contiguous()) # torch.Size([730, 17, 24]), False
   
    # daily_bikes was transp., This should be shaped (B, C, L), cols. in [1]
    daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, 
                                        daily_bikes.shape[2])
    print(daily_weather_onehot.shape)   # torch.Size([730, 4, 24])

# Why do we unsqueeze here if same dim? Throws error 
# ANSWER: Since we are picking a col., we lose a dim.    

# RuntimeError: Index tensor must have the same number of dimensions as self tensor
#    daily_weather_onehot.scatter_(
#        dim=1,
#        index=daily_bikes[:,9,:].long(),
#        value=1.0
#    )
#    print(daily_weather_onehot.shape)

    if(False):
        print('\n\nTemp ex:')
        temp = daily_bikes[:,9,:]   
        print(temp.shape)   # torch.Size([730, 24])
    # ANSWER: Since we are picking a col., we lose a dim.     


    daily_weather_onehot.scatter_(
        dim=1,
        index=daily_bikes[:,9,:].long().unsqueeze(1) - 1, # same reason,-1 for weather var
        value=1.0
    )
    print(daily_weather_onehot.shape)   # torch.Size([730, 4, 24])
#    print(daily_weather_onehot[1,:,:])

    print(daily_bikes.shape) # torch.Size([730, 17, 24])
    daily_bikes = torch.cat((daily_bikes, daily_weather_onehot),dim=1)
    print(daily_bikes.shape) # torch.Size([730, 21, 24])


    print('End ex4') 


if __name__ == "__main__":
    ex4()








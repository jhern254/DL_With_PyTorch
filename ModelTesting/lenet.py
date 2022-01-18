import torch
import torch.nn as nn # All NN modules, nn.Linear, nn.Conv2d, BatchNorm, Loss fns
import logging

logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.DEBUG)

# LeNet Architecture
# Made for MNIST data set, B x 1 x 28 x 28 imgs
# 1x32x32 Input -> (5x5),s=1,p=0 -> avg. pool s=2,p=0 -> (5x5),s=1,p=0 ->
# avg. pool s=2,p=0 -> Conv 5x5 to 120 channels -> LinearFC 120 -> 
# 84 x LinearFC 10. 
# Output tensor: B x 10

class LeNet(nn.Module):
    """
        Padding is 0 for now since just testing w/ random tensor
    """
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,
                                stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.pool(self.act(self.conv1(x))) # layer 1
        out = self.pool(self.act(self.conv2(out))) # layer 2
        out = self.act(self.conv3(out)) # layer 3, num_examples x 120 x 1 x 1
        # reshape tensor for fc linear layer -> num_examples x 120
        out = out.view(-1, 120)
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        return out



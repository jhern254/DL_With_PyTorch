import math

from torch import nn as nn
import torch.nn.functional as F

from logconf import logging

log = logging. getLogger(__name__)
log.setLevel(logging.DEBUG)


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super(LunaBlock, self).__init__

        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        # self conv. layer
        self.conv2 = nn.Conv3d(             
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )


    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        # do operations in place
        block_out = F.relu(block_out, inplace=True)
        block_out = self.conv2(block_out)
        block_out = F.ReLU(block_out, inplace=True)







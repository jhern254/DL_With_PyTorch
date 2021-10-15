import argparse
import datetime
import os
import sys

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

#from util.util import enumerateWithEstimate
from dsets import LunaDataset
from logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class LunaTrainingApp:
    def __init(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]     # get args from CLI

        # set up CL args for app
        parser = argparse.ArgumentParser()
        paser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )
        # Tensorboard CL args
        parser.add_argument('--tb-prefix',
            default='p2ch11',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('commment',
            help="Comment suffix for Tensorboard run."
            nargs='?',
            default='dwlpt',
        )
        # set member vars
        self.cli_args = parser.parse_args(sys_argv) # cli_args is a list w/ each arg.
        self.time_str = datetime.datetime.now().strftime('%y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        # TODO: More code
        # ...


        # initialize model and optimizer through method fns
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # NOTE: Always init model 1st, optim. needs to know if GPU or CPU
        self.model = self.initModel()
        self.optimizer = self.initOptimizer

    def initModel(self):
        """
        Set model and use CUDA 
        """
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        # from book code, has optional ADAM commented out
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initTrainDl(self):
        """
            Train loader class from prev. ch. 
        """
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        # extend our custom class for batch size arg
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        # Same as train but use val set
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        
        return val_dl

# TODO: Tensorboard fns


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()










if __name__ == '__main__':
    LunaTrainingApp().main()



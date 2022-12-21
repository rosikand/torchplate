"""
File: experiments.py
------------------
This file holds the experiments which are
subclasses of torchplate.experiment.Experiment. 
"""

import numpy as np
import torchplate
from torchplate import (
        experiment,
        utils
    )
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models
import datasets



class BaseExp(experiment.Experiment):
    def __init__(self): 
        self.model = models.CifarMLP()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.trainloader, self.testloader = datasets.get_dataloaders(path="https://stanford.edu/~rsikand/assets/datasets/mini_cifar.pkl")
        

        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = None,
            verbose = True
        )
    
    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        pass 


    def test(self):
        pass



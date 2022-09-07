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
        self.trainloader, self.testloader = datasets.get_dataloaders(
            path="https://stanford.edu/~rsikand/assets/datasets/mini_cifar.pkl", 
            DatasetClass=datasets.BaseDataset
            )
        #self.run = wandb.init(project="yourproject", entity="yourname")

        # inherit from torchplate.experiment.Experiment and pass in
        # model, optimizer, and dataloader 
        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = None,
            verbose = True
        )
    
    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        x, y = batch
        logits = self.model(x)
        loss_val = self.criterion(logits, y)
        return loss_val

    def test(self):
        accuracy_count = 0
        for x, y in self.testloader:
            logits = self.model(x)
            pred = torch.argmax(F.softmax(logits, dim=1)).item()
            print(f"Prediction: {pred}, True: {y.item()}")
            if pred == y:
                accuracy_count += 1
        print("Accuracy: ", accuracy_count/len(self.testloader))
        #self.run.finish()



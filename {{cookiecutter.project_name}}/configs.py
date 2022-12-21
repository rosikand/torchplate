"""
File: configs.py 
----------------------
Specifies config parameters. 
"""


import experiments
import torchplate
import rsbox 
import wandb
from rsbox import ml, misc
import torch.optim as optim
import torch
from torch import nn
import pickle 
import timm



class BaseConfig:
    experiment = experiments.BaseExp
    trainloader, test_set = data.get_dataloaders()
    logger = None
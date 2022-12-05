"""
File: configs.py 
----------------------
Specifies config parameters. 
"""


import datasets 
import models
import experiments
import torchplate
import rsbox 
from rsbox import ml, misc
import torch.optim as optim
import segmentation_models_pytorch as smp


class BaseConfig:
    experiment = experiments.BaseExp
    dataset_dist = misc.load_dataset("https://stanford.edu/~rsikand/assets/datasets/mini_cell_segmentation.pkl")
    trainloader, testloader = torchplate.utils.get_xy_loaders(dataset_dist)
    model_class = models.SmpUnet(
        encoder_name='resnet34', 
        encoder_weights='imagenet', 
        classes=1, 
        in_channels=3,
        activation='sigmoid'
    )
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = optim.Adam(model_class.model.parameters(), lr=0.001)

    
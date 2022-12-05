"""
File: models.py
------------------
This file holds the torch.nn modules. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchplate
from torchplate import (
    experiment,
    utils
)
import segmentation_models_pytorch as smp




class SmpUnet(utils.BaseModelInterface):
    # note: each child should provide a dict containing the relevant model params if needed 
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', classes=1, in_channels=1, activation='sigmoid'):
        # model
        self.model = smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights, 
            classes=classes, 
            in_channels=in_channels,
            activation=activation
        )

        # preprocessing
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            encoder_name=encoder_name, 
            pretrained=encoder_weights
        )
        
        super().__init__(
            model = self.model
        )

    
    def preprocess(self, inputs):
        # input should be of shape (n, c, h, w)
        if self.preprocessing_fn is not None:
            # preprocess input
            inputs = torch.movedim(inputs, 1, -1)  # (n, c, h, w) --> (n, h, w, c) 
            inputs = self.preprocessing_fn(inputs)
            inputs = torch.movedim(inputs, -1, 1)  # (n, h, w, c) --> (n, c, h, w) 
        inputs = inputs.to(torch.float)
        return inputs

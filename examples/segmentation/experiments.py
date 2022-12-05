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
import pdb
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models
import segmentation_models_pytorch as smp



class BaseExp(experiment.Experiment):
    def __init__(self, config): 
        self.cfg = config
        self.model_class = self.cfg.model_class
        self.model = self.model_class.model
        self.trainloader = self.cfg.trainloader
        self.testloader = self.cfg.testloader
        self.criterion = self.cfg.loss_fn
        self.optimizer = self.cfg.optimizer
        
       
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
        logits = self.model_class.forward_pipeline(x)
       #  y_un_one_hot = torch.argmax(y, dim=1)
        y_un_one_hot = y[:, 1, :, :]
        loss_val = self.criterion(logits, y_un_one_hot)
        return loss_val

    
    @staticmethod
    def compute_iou(logits, y):
        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(logits, y.long(), mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        return iou_score 


    def test(self):
        # test the model on the test set
        iou_scores = []
        with torch.no_grad():
            for batch in self.testloader:
                x, y = batch
                logits = self.model_class.forward_pipeline(x)
                y_un_one_hot = y[:, 1:, :, :]
                iou_score = self.compute_iou(logits, y_un_one_hot)
                iou_scores.append(iou_score)

        iou_score_avg = np.mean(np.array(iou_scores))
        print(f"Average IoU score (testset): {iou_score_avg}")

# Segmentation  

This page showcases an example using torchplate to train a visual segmentation model. The program is divided into several `.py` files. 

## `configs.py`

```python
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

    
```

## `experiments.py`

```python
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
```

## `models.py` 


```python
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
```

## `runner.py` 

```python
"""
File: runner.py
------------------
Runner script to train the model. This is the script that calls the other modules.
Execute this one to execute the program! 
"""


import configs
import argparse
import warnings
import pdb 
import experiments


def main(args):
    if args.config is None:
        config_class = 'BaseConfig'
    else:
        config_class = args.config
    cfg = getattr(configs, config_class)
    exp = cfg.experiment(
        config=cfg
    )

	# train the model
    exp.test() 
    exp.train(num_epochs=15)
    exp.test()



if __name__ == '__main__':
    # configure args 
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-config", type=str, help='specify config.py class to use.') 
    args = parser.parse_args()
    main(args)
	
```


## Output 

Run: 

```
$ python3 run.py -c BaseConfig
```

Output: 

```
Average IoU score (testset): 0.03814944624900818
Epoch 1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:06<00:00,  1.79it/s]
Training Loss (epoch 1): 0.6883486130020835
Epoch 2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.87it/s]
Training Loss (epoch 2): 0.6678070100871
Epoch 3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.87it/s]
Training Loss (epoch 3): 0.6593567999926481
Epoch 4: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.87it/s]
Training Loss (epoch 4): 0.6544880541888151
Epoch 5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.87it/s]
Training Loss (epoch 5): 0.6512440117922697
Epoch 6: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.87it/s]
Training Loss (epoch 6): 0.6492173671722412
Epoch 7: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.87it/s]
Training Loss (epoch 7): 0.6479388150301847
Epoch 8: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.87it/s]
Training Loss (epoch 8): 0.6467715881087563
Epoch 9: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.88it/s]
Training Loss (epoch 9): 0.6455191428011114
Epoch 10: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.85it/s]
Training Loss (epoch 10): 0.6443771774118597
Epoch 11: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.84it/s]
Training Loss (epoch 11): 0.6441563692959872
Epoch 12: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.86it/s]
Training Loss (epoch 12): 0.6434326388619163
Epoch 13: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.86it/s]
Training Loss (epoch 13): 0.6425378214229237
Epoch 14: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:06<00:00,  1.72it/s]
Training Loss (epoch 14): 0.6431863037022677
Epoch 15: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:05<00:00,  1.85it/s]
Training Loss (epoch 15): 0.6430070671168241
Finished Training!
Average IoU score (testset): 0.10699300467967987
```
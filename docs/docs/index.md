# `torchplate`: Minimal Experiment Workflows in PyTorch 

([Github](https://github.com/rosikand/torchplate) | [PyPI](https://pypi.org/project/torchplate))


An extremely minimal and simple experiment module for machine learning in PyTorch. 

In addition to abstracting away the training loop, we provide several abstractions to improve the efficiency of machine learning workflows with PyTorch. 


## Example 

To get started, create an experiment child class of `torchplate.experiment.Experiment` and provide several key, experiment-unique items: model, optimizer, and a training set dataloader. Add whatever custom methods you may want to this class. Then starting training! That's it! 

```python
import torchplate
import data 
import models
import torch
import torch.optim as optim
import torch.nn as nn


class SampleExp(torchplate.experiment.Experiment):
    def __init__(self): 
        self.model = models.Net()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        dataset = data.load_set('cifar')
        # use various torchplate.utils to improve efficiency of common workflows 
        self.trainloader, self.testloader = torchplate.utils.get_xy_loaders(dataset)

        # inherit from torchplate.experiment.Experiment and pass in
        # model, optimizer, and dataloader 
        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader 
        )
    
    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        x, y = batch
        logits = self.model(x)
        loss_val = self.criterion(logits, y)
        return loss_val


exp = SampleExp()
exp.train(num_epochs=5)
```
output: 
```
Epoch 1: 100%|███████████████████████████████████████████████████████████| 27/27 [00:00<00:00, 293.98it/s]
Training Loss (epoch 1): 1.3564644632516083
Epoch 2: 100%|███████████████████████████████████████████████████████████| 27/27 [00:00<00:00, 598.46it/s]
Training Loss (epoch 2): 1.2066593832439847
Epoch 3: 100%|███████████████████████████████████████████████████████████| 27/27 [00:00<00:00, 579.40it/s]
Training Loss (epoch 3): 1.1030386642173484
Epoch 4: 100%|███████████████████████████████████████████████████████████| 27/27 [00:00<00:00, 563.90it/s]
Training Loss (epoch 4): 1.0885229706764221
Epoch 5: 100%|███████████████████████████████████████████████████████████| 27/27 [00:00<00:00, 577.54it/s]
Training Loss (epoch 5): 1.0520343957123932
Finished Training!
```

## Installation 

```
$ pip install torchplate
```


## Changelog 

### 0.0.1

- First version published. Provides basic data-loading utilities and the base experiment module. 

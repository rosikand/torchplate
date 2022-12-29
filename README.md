# `torchplate`: Minimal Experiment Workflows in PyTorch 

([Github](https://github.com/rosikand/torchplate) | [PyPI](https://pypi.org/project/torchplate) | [Documentation](https://rosikand.github.io/torchplate/))

[Installation](#installation) | [Example](#example) | [More examples](#more-examples) | [Starter project](#starter-project) | [Changelog](#changelog)

An extremely minimal and simple experiment module for machine learning in PyTorch (PyTorch + boilerplate = `torchplate`).

In addition to abstracting away the training loop, we provide several abstractions to improve the efficiency of machine learning workflows with PyTorch. 

## Installation 

```
$ pip install torchplate
```

## Example 

To get started, create a child class of `torchplate.experiment.Experiment` and provide several key, experiment-unique items: model, optimizer, and a training set dataloader. Then, provide an implementation of the abstract method `evaluate`. This function takes in a batch from the `trainloader` and should return the loss (i.e., implement the forward pass + loss calculation). Add whatever custom methods you may want to this class. Then starting training! That's it! 

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
exp.train(num_epochs=10, gradient_accumulate_every_n_batches=4, display_batch_loss=False)
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

### More examples

See `examples/cifar` for another minimal example. See `examples/starter` for a full program example. To get started running your own experiments, you can use `examples/starter` as a base (or use cookiecutter as shown below). 

### Starter project 

The `starter` branch holds the source for a [cookiecutter](https://github.com/cookiecutter/cookiecutter) project. This allows users to easily create projects from the starter code example by running a simple command. To get started, [install cookiecutter](https://cookiecutter.readthedocs.io/en/stable/installation.html#install-cookiecutter) and then type
```
$ cookiecutter https://github.com/rosikand/torchplate.git --checkout starter
```

which will generate the following structure for you to use as a base for your projects: 

```
torchplate_starter
├── datasets.py
├── experiments.py
├── models.py
└── runner.py
```


## Changelog 

### 0.0.7

- Largest change to date. New features: gradient accumulation, save weights every $n$ epochs, display batch loss, metrics, metrics interfaced with `train`. 

### 0.0.6

- Fixed bug in model weight saving. 

### 0.0.5
- Added model weights loading and saving. 

### 0.0.4 
- Several changes: added callbacks, changed verbose default to true, added `ModelInterface` pipeline to `utils`. 

### 0.0.3
- Added verbose option as well as wandb logging 

### 0.0.2
- Fixed a polymorphic bug 

### 0.0.1
- First version published. Provides basic data-loading utilities and the base experiment module. 



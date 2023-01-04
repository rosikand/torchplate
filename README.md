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
from torchplate import experiment
from torchplate import utils
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import requests
import cloudpickle as cp
from urllib.request import urlopen


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3*32*32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CifarExp(torchplate.experiment.Experiment):
    def __init__(self): 
        self.model = Net()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        dataset = cp.load(urlopen("https://stanford.edu/~rsikand/assets/datasets/mini_cifar.pkl")) 
        # use various torchplate.utils to improve efficiency of common workflows 
        self.trainloader, self.testloader = torchplate.utils.get_xy_loaders(dataset)

        # inherit from torchplate.experiment.Experiment and pass in
        # model, optimizer, and dataloader 
        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
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

    def on_epoch_end(self):
        # to illustrate the concept of callbacks 
        print("------------------ (Epoch end) --------------------")



exp = CifarExp()
exp.train(num_epochs=100)
exp.test()
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

### 0.0.8

- Fixed `metrics` import bug. 


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



"""
File: experiment.py
------------------
Provides the main module of the package: Experiment. 
"""

from abc import ABC, abstractmethod
import torch 
from tqdm.auto import tqdm
import wandb 


class Experiment(ABC):
    """
    Base experiment superclass. All other experiments
    should inherit from this class. Each sub-experiment
    must provide an implementation of the "evaluate" abstract
    method. A sub-experiment has full autonomy to override
    the basic components such as the training loop "train". 
    """
    def __init__(self, model, optimizer, trainloader, wandb_logger=None, verbose=False):
        """
        Experiment superclass initializer. Each subclass must provide
        a model, optimizer, and trainloader at the very least. 
        Arguments: 
        -----------
        - model: torch nn.module 
        - optimizer: torch optimizer 
        - trainloader: torch Dataloader to be used for training 
        Optional:
        - wandb_logger (wandb.init object): pass in if you want to log to wandb. Default: None. 
        - verbose (boolean): if true, print out metrics during training. Default: False. 
        """
        self.model = model 
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.wandb_logger = wandb_logger
        self.verbose = verbose


    def train(self, num_epochs):
        """
        Training loop. 
        """
        self.model.train()
        epoch_num = 0

        for epoch in range(num_epochs):  # loop over the dataset num_epochs times 
            epoch_num += 1
            running_loss = 0.0
            
            tqdm_loader = tqdm(self.trainloader)
            for batch in tqdm_loader:
                tqdm_loader.set_description(f"Epoch {epoch_num}")
                self.optimizer.zero_grad()
                loss = self.evaluate(batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() 

            epoch_avg_loss = running_loss/len(self.trainloader)
            if self.wandb_logger is not None:
                self.wandb_logger.log({"Training loss": epoch_avg_loss})
            if self.verbose:
                print("Training Loss (epoch " + str(epoch_num) + "):", epoch_avg_loss)
        
        self.model.eval()
        print('Finished Training!')


    @abstractmethod
    def evaluate(self, batch):
        """
        Abstract method which the user must provide. Implement the 
        forward pass and return the loss value. 
        Arguments:
        -----------
        - batch: a batch from the train data loader (i.e., an (x, y) pair). To
        be used as input into the model. 
        Returns:
        -----------
        - A scalar loss value. 
        """
        pass

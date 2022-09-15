"""
File: experiment.py
------------------
Provides the main module of the package: Experiment. 
"""

from abc import ABC, abstractmethod
import torch 
import os
import rsbox
from rsbox import misc
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
    def __init__(self, model, optimizer, trainloader, wandb_logger=None, verbose=True):
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
        - verbose (boolean): if true, print out metrics during training. Default: True. 
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
        self.on_run_start()
        self.model.train()
        epoch_num = 0

        for epoch in range(num_epochs):
            epoch_num += 1
            self.on_epoch_start()
            running_loss = 0.0
            
            tqdm_loader = tqdm(self.trainloader)
            for batch in tqdm_loader:
                tqdm_loader.set_description(f"Epoch {epoch_num}")
                self.on_batch_start()
                self.optimizer.zero_grad()
                loss = self.evaluate(batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() 
                self.on_batch_end()

            epoch_avg_loss = running_loss/len(self.trainloader)
            if self.wandb_logger is not None:
                self.wandb_logger.log({"Training loss": epoch_avg_loss})
            if self.verbose:
                print("Training Loss (epoch " + str(epoch_num) + "):", epoch_avg_loss)
            self.on_epoch_end()
        
        self.model.eval()
        print('Finished Training!')
        self.on_run_end()


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


    def on_batch_start(self):
        """
        Callback that can be overriden. Implement whatever you want
        to happen before each batch iteration. 
        """
        pass


    def on_batch_end(self):
        """
        Callback that can be overriden. Implement whatever you want
        to happen after each batch iteration. 
        """
        pass


    def on_epoch_start(self):
        """
        Callback that can be overriden. Implement whatever you want
        to happen before each epoch iteration. 
        """
        pass


    def on_epoch_end(self):
        """
        Callback that can be overriden. Implement whatever you want
        to happen after each epoch iteration. 
        """
        pass


    def on_run_start(self):
        """
        Callback that can be overriden. Implement whatever you want
        to happen before each run. 
        """
        pass


    def on_run_end(self):
        """
        Callback that can be overriden. Implement whatever you want
        to happen after each run. 
        """
        pass
    
    
    def save_weights(self, save_path=None):
        """
        Function to save model weights at 'save_path'. 
        Arguments:
        - save_path: path to save the weights. If not given, defaults to current timestamp. 
        """ 
        if save_path is None:
            if not os.path.exists("saved"):
                os.makedirs("saved")
            save_path = "saved/" + misc.timestamp() + ".pth"
        torch.save(self.model.state_dict(), save_path)
        print("Model weights saved at: " + str(save_path))
        
        
    def load_weights(self, weight_path):
        """
        Function to load model weights saved at 'weight_path'. 
        Arguments:
        - weight_path: path pointing to the saved weights. 
        """
        self.model.load_state_dict(torch.load(weight_path))
        print("Weights loaded!")

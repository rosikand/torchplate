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
from torchplate import metrics


class Experiment(ABC):
    """
    Base experiment superclass. All other experiments
    should inherit from this class. Each sub-experiment
    must provide an implementation of the "evaluate" abstract
    method. A sub-experiment has full autonomy to override
    the basic components such as the training loop "train". 
    """
    def __init__(self, model, optimizer, trainloader, save_weights_every_n_epochs=None, wandb_logger=None, verbose=True, experiment_name=misc.timestamp()):
        """
        Experiment superclass initializer. Each subclass must provide
        a model, optimizer, and trainloader at the very least. 
        Arguments: 
        -----------
        - model: torch nn.module 
        - optimizer: torch optimizer 
        - trainloader: torch Dataloader to be used for training 
        Optional:
        - save_weights_every_n_epochs: how often to save the model weight automatically. Default: None.
            Specify None if you don't want to save weights automatically. 
        - wandb_logger (wandb.init object): pass in if you want to log to wandb. Default: None. 
        - verbose (boolean): if true, print out metrics during training. Default: True. 
        - experiment_name (str): name of the experiment for saving. Default: timestamp. 
        """
        self.model = model 
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.wandb_logger = wandb_logger
        self.verbose = verbose
        self.save_weights_every_n_epochs = save_weights_every_n_epochs
        assert type(self.save_weights_every_n_epochs) is int or self.save_weights_every_n_epochs is None, "save_weights_every_n_epochs must be an integer or None"
        self.epoch_num = 0
        self.experiment_name = experiment_name
        assert type(self.experiment_name) is str, "experiment name must be a string"


    def train(self, num_epochs, gradient_accumulate_every_n_batches=1, display_batch_loss=False):
        """
        Training loop. Can optionally specify how often to accumulate gradients. Default: 1. 
        """
        self.on_run_start()
        self.model.train()

        metrics_ = {}
        first_batch = True
        eval_returns_loss_only = False

        history = {}

        for epoch in range(num_epochs):
            self.epoch_num += 1
            self.on_epoch_start()
            running_loss = 0.0
            
            tqdm_loader = tqdm(self.trainloader)
            batch_idx = -1
            for batch in tqdm_loader:
                batch_idx += 1
                if not display_batch_loss:
                    tqdm_loader.set_description(f"Epoch {self.epoch_num}")
                self.on_batch_start()
                
                evals = self.evaluate(batch)

                # registration of metrics
                if first_batch:
                    if torch.is_tensor(evals):
                        if evals.dim() == 0:
                            eval_returns_loss_only = True
                            history["loss"] = []
                    else:
                        assert type(evals) is dict, "if you aren't providing a scalar loss value in evaluate, you must return a dictionary."
                        assert "loss" in evals.keys(), "evaluate must return a 'loss' value"
                        # register metrics
                        for key in evals:
                            curr_metric = metrics.MeanMetric()
                            metrics_[key] = curr_metric
                            history[key] = []
                    first_batch = False

                # get loss val 
                if eval_returns_loss_only: 
                    loss = evals
                else:
                    loss = evals["loss"]

                if display_batch_loss:
                    tqdm_loader.set_description(f"Epoch {self.epoch_num} | loss: {loss:.4f}")

                loss.backward()


                # gradient accumulation
                if gradient_accumulate_every_n_batches > 1:
                    loss = loss / gradient_accumulate_every_n_batches
                
                    if ((batch_idx + 1) % gradient_accumulate_every_n_batches == 0) or (batch_idx + 1 == len(self.trainloader)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()


                # per-batch updates
                if not eval_returns_loss_only:
                    # update metrics 
                    for key in metrics_:
                        metrics_[key].update(evals[key])

                running_loss += loss.item() 
                self.on_batch_end()

            # per-epoch updates 

            epoch_avg_loss = running_loss/len(self.trainloader)
            if self.wandb_logger is not None:
                if eval_returns_loss_only:
                    self.wandb_logger.log({"Training loss": epoch_avg_loss})
                else:
                    self.wandb_logger.log(metrics_)
            if self.verbose:
                if eval_returns_loss_only:
                    print("Training Loss (epoch " + str(self.epoch_num) + "):", epoch_avg_loss)
                else:
                    for key in metrics_:
                        print(f"Training {key} (epoch {str(self.epoch_num)}): {metrics_[key].get()}")
            
            # reset metrics and update history 
            if not eval_returns_loss_only:
                for key in metrics_:
                    append_val = metrics_[key].get()
                    if torch.is_tensor(append_val):
                        if append_val.requires_grad:
                            append_val = append_val.detach().cpu().item()
                    history[key].append(append_val)
                    metrics_[key].reset()
            else:
                history["loss"].append(epoch_avg_loss)


            # weight saving 
            if self.save_weights_every_n_epochs is not None:
                if self.epoch_num % self.save_weights_every_n_epochs == 0:
                    if not os.path.exists("saved"):
                        os.makedirs("saved")
                    save_path = "saved/epoch_" + str(self.epoch_num) + "-" + self.experiment_name
                    self.save_weights(save_path)


            self.on_epoch_end()
        
        self.model.eval()
        print('Finished Training!')
        self.epoch_num = 0
        self.on_run_end()

        return history


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

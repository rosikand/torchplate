"""
File: utils.py
------------------
We also provide some experiment workflow
utilities which one can use and import from
this module. 
"""

import torch 
from torch.utils.data import Dataset
import os
import rsbox 
from rsbox import misc


class BaseModelInterface:
    """
    Wrapper class which provides a model
    interface for torch.nn models. Mainly,
    this class provides the forward pass pipeline
    function, 'predict' which sends an 
    input through this pipeline:
        preprocess --> model --> postprocess. 
    Users must provide a torch.nn model and can
    optionally specify preprocess and postprocess
    functions. 
    """
    def __init__(self, model):
        """
        Provide torch.nn module. 
        """
        self.model

    def preprocess(self, inputs):
        return inputs

    def postprocess(self, inputs):
        return inputs

    def forward_pipeline(self, inputs):
        preprocessed_inputs = self.preprocess(inputs)
        logits = self.model(preprocessed_inputs)
        processed_output = self.postprocess(logits)
        return processed_output
    
    def save_weights(self, save_path=None):
        if save_path is None:
            if not os.path.exists("saved"):
                os.makedirs("saved")
            save_path = "saved/" + misc.timestamp() + ".pth"
        torch.save(self.model.state_dict(), save_path)
        print("Model weights saved at: " + str(save_path))
        
    def load_weights(self, weight_path):
        self.model.load_state_dict(torch.load(weight_path))
        print("weights loaded!")


class XYDataset(Dataset):
    """
    PyTorch Dataset class for datasets of the 
    form [(x,y), ..., (x,y)]. 
    """
    def __init__(self, data_set):
        """
        Arguments:
        -----------
        - distribution (sequence): sequence of the form [(x,y), ..., (x,y)]
        representing the dataset. 
        """
        self.data_distribution = data_set
        
    def __getitem__(self, index):
        sample = self.data_distribution[index][0]
        label = self.data_distribution[index][1]
        sample = torch.tensor(sample, dtype=torch.float)
        label = torch.tensor(label)
        return (sample, label) 
        
    def __len__(self):
        return len(self.data_distribution)


def get_xy_dataset(distribution):
    """
    Given a dataset of the form [(x,y), ..., (x,y)],
    returns a PyTorch Dataset object. 
    Arguments:
    -----------
    - distribution (sequence): sequence of the form [(x,y), ..., (x,y)]
    representing the dataset. 
    Returns:
    -----------
    - a torch.utils.data.Dataset object 
    """
    return XYDataset(distribution)


def split_dataset(torch_set, ratio=0.9):
    """
    Given a torch.utils.data.Dataset object, this function splits it 
    into train and test a torch.utils.data.Dataset objects. 
    The split is random is the size is based on the input ratio.  
    Arguments:
    -----------
    - torch_set: a torch.utils.data.Dataset object containing the entire dataset 
    - ratio: train/test ratio split. Default is 0.9. 
    Returns:
    -----------
    Tuple consisting of: 
    - trainset: a torch.utils.data.Dataset object to be used for training 
    - testset: a torch.utils.data.Dataset object to be used for testing 
    """
    train_size = int(ratio * len(torch_set))  
    test_size = len(torch_set) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(torch_set, [train_size, test_size])
    return train_dataset, test_dataset


def get_loaders(torch_sets):
    """
    Given a sequence of torch.utils.data.Dataset objects, this function
    wraps them all in torch.utils.data.Dataloader objects and returns 
    a sequence in the same order. Note that this function doesn't
    support custom arguments to the torch.utils.data.DataLoader call.
    If one desires to use custom arguments (e.g., batch_size), they should
    call torch.utils.data.DataLoader themselves. 
    Arguments:
    -----------
    - torch_sets (sequence): a sequence consisting of a torch.utils.data.Dataset objects. 
    Returns:
    -----------
    - loaders (sequence): the datasets wrapped in a torch.utils.data.Dataloader objects 
    (returned in the same order.)
    """

    loaders = []

    for torch_set in torch_sets:
        current_set = torch.utils.data.DataLoader(torch_set)
        loaders.append(current_set)

    return loaders


def get_xy_loaders(distribution):
    """
    end-to-end function which returns train and test loaders
    given a sequence of the form [(x,y), ..., (x,y)]. If more customization
    is needed, please call the other utility functions individually. 
    Arguments:
    -----------
    - distribution (sequence): dataset of the form [(x,y), ..., (x,y)]. 
    Returns:
    -----------
    - loaders (sequence): the datasets wrapped in a torch.utils.data.Dataloader objects 
    (returned in the same order). 
    """
    torch_set = get_xy_dataset(distribution)
    torch_sets = split_dataset(torch_set)
    loaders = get_loaders(torch_sets)
    trainloader = loaders[0]
    testloader = loaders[1]

    return trainloader, testloader

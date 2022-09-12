"""
File: datasets.py
------------------
This file holds various dataset and dataloading
functions. 
"""

import cloudpickle as cp
import torch
from torch.utils.data import Dataset
import torchplate
from torchplate import utils as tp_utils
import requests 
from urllib.request import urlopen



class BaseDataset(Dataset):
    def __init__(self, data_set):
        self.data_distribution = data_set
        
    def __getitem__(self, index):
        sample = self.data_distribution[index % len(self.data_distribution)][0]
        label = self.data_distribution[index % len(self.data_distribution)][1]
        sample = torch.tensor(sample, dtype=torch.float)
        label = torch.tensor(label)
        return (sample, label) 
        
    def __len__(self):
        return len(self.data_distribution)


def get_dataloaders(path, DatasetClass):
	data_distribution = cp.load(urlopen(path))
	torch_set = DatasetClass(data_distribution)
	train_dataset, test_dataset = tp_utils.split_dataset(torch_set, ratio=0.9)
	trainloader = torch.utils.data.DataLoader(train_dataset)
	testloader = torch.utils.data.DataLoader(test_dataset)
	return trainloader, testloader

"""
File: metrics.py
------------------
Metric classes for evaluating model performance. 
"""


import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class MeanMetric:
  """
  Scalar metric designed to use on a per-epoch basis 
  and updated on per-batch basis. For getting average across
  the epoch. 
  """

  def __init__(self):
    self.vals = []

  def update(self, new_val):
    self.vals.append(new_val)

  def reset(self):
    self.vals = []

  def get(self):
    mean_value = sum(self.vals)/len(self.vals)
    return mean_value


class Accuracy(MeanMetric):
    """
    Subclass of MeanMetric which defines 
    a standard accuracy update function. 
    """

    def update(self, logits, labels):
        return calculate_accuracy(logits, labels)
    

class MeanMetricCustom(ABC, MeanMetric):
  """
  Abstract scalar metric. Must provide calculation given logits and y. 
  """

  def __init__(self):
    self.vals = []
  
  @abstractmethod
  def calculate(self, logits, y):
    # returns a value
    pass
    
  def update(self, logits, y):    
    self.vals.append(self.calculate(logits,y))



# --------------------------- (scores) ------------------------------


def calculate_accuracy(logits, y):
    assert len(logits.shape) == 2 and len(y.shape) == 1
    assert logits.shape[0] == y.shape[0]
    probs = F.softmax(logits, dim=1)
    y = torch.argmax(probs, dim=-1) == y
    y = y.type(torch.float)
    return torch.mean(y).item()

from datetime import datetime
from time import time

import numpy as np
import torch


def human_readable_timestamp():
    """
    Return current time in a nice format for logging purposes
    """
    return datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S')


def accuracy(pred_x: torch.Tensor, true_x: torch.Tensor):
    """
    Classification accuracy

    :param pred_x: Predicted tensor of shape (n_batch, n_classes)
    :param true_x: True tensor of shape (n_batch,)
    :return accuracy: Percentage of corrected predictions 
    """
    _, max_indices = torch.max(pred_x, 1)
    assert max_indices.size() == true_x.size()
    n_corrects = (max_indices == true_x).sum().item()
    return n_corrects / pred_x.size(0) * 100


def invert_permutation(p: np.ndarray):
    """
    Return array of indexes to invert permutation `p`.

    :param p: permutation of 0, 1, ..., len(p)-1.
    :return p_inv: array where `p_inv[i]` gives the index of `i` in `p`. 
    """
    p_inv = np.empty_like(p)
    p_inv[p] = np.arange(p.size)
    return p_inv

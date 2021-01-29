import numpy as np
import torch

def mse(y, outs):
    ans = ((y - outs)**2).mean()
    return ans.item()

def rmse(y, outs):
    ans = np.sqrt(mse(y, outs))
    return ans

def mae(y, outs):
    ans = (torch.abs(y - outs)).mean()
    return ans.item()

def accuracy(y, outs):
    ans = (y == torch.round(outs)).float().mean() * 100
    return ans.item()

def examples(y, outs):
    idx = np.random.choice(y.shape[0], 10, replace=False)
    return y[idx], outs[idx]



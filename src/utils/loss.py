import torch

def mse_loss(pred, true):
    return torch.mean((pred - true) ** 2)

def l1_loss(pred, true):
    return torch.mean(torch.abs(pred - true))
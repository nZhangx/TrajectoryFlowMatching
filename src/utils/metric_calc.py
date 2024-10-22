import torch.nn as nn
import torch
import numpy as np
# dist calculations
import math
from typing import Union

import numpy as np
import torch

from typing import Optional
from .mmd import linear_mmd2, mix_rbf_mmd2, poly_mmd2
# from .optimal_transport import wasserstein
import ot as pot
from functools import partial
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
import pandas as pd


# auroc and auprc
from sklearn.metrics import roc_auc_score, average_precision_score



def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret

def compute_distances(pred, true):
    """computes distances between vectors."""
    mse = torch.nn.functional.mse_loss(pred, true).item()
    me = math.sqrt(mse)
    return mse, me, torch.nn.functional.l1_loss(pred, true).item()

def compute_distribution_distances_new(pred: torch.Tensor, true: Union[torch.Tensor, list]):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.
    return the eval for the last time point
    """
    NAMES = [
        "1-Wasserstein",
        "2-Wasserstein",
        "Linear_MMD",
        "Poly_MMD",
        "RBF_MMD",
        "Mean_MSE",
        "Mean_L2",
        "Mean_L1",
        "Median_MSE",
        "Median_L2",
        "Median_L1",
    ]
    is_jagged = isinstance(true, list)
    pred_is_jagged = isinstance(pred, list)
    dists = []
    to_return = []
    names = []
    filtered_names = [name for name in NAMES if not is_jagged or not name.endswith("MMD")]
    ts = len(pred) if pred_is_jagged else pred.shape[1]
    # for t in np.arange(ts):
    t = max(ts - 1, 0)
    if pred_is_jagged:
        a = pred[t]
    else:
        a = torch.tensor(pred).float().clone().detach()
    if is_jagged:
        b = true[t]
    else:
        b = torch.tensor(true).float().clone().detach()
    w1 = wasserstein(a, b, power=1)
    w2 = wasserstein(a, b, power=2)

    if not pred_is_jagged and not is_jagged:
        mmd_linear = linear_mmd2(a, b).item()
        mmd_poly = poly_mmd2(a, b, d=2, alpha=1.0, c=2.0).item()
        mmd_rbf = mix_rbf_mmd2(a, b, sigma_list=[0.01, 0.1, 1, 10, 100]).item()
    mean_dists = compute_distances(torch.mean(a, dim=0), torch.mean(b, dim=0))
    median_dists = compute_distances(torch.median(a, dim=0)[0], torch.median(b, dim=0)[0])

    if pred_is_jagged or is_jagged:
        dists.append((w1, w2, *mean_dists, *median_dists))
    else:
        dists.append((w1, w2, mmd_linear, mmd_poly, mmd_rbf, *mean_dists, *median_dists))

    to_return.extend(np.array(dists).mean(axis=0))
    names.extend(filtered_names)
    return names, to_return


def metrics_calculation(pred, true, metrics=['mse_loss', 'l1_loss'], cutoff=-0.91, map_idx = 1):
    
    # if pred is a tensor, convert to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().squeeze().numpy()
        true = true.detach().cpu().squeeze().numpy()

    loss_D = {key : None for key in metrics}
    for metric in metrics:
        if metric == 'mse_loss':
            loss_D['mse_loss'] = np.mean((pred - true)**2)
            # self.log('mse_loss', self.loss_fn(pred, true))
        if metric == 'l1_loss':
            loss_D['l1_loss'] = np.mean(np.abs(pred - true))
            # self.log('l1_loss', torch.mean(torch.abs(pred - true)))
        if metric == 'crit_map':
            auroc, auprc = critical_state_pred(pred, true, cutoff, map_idx)
            loss_D['crit_state_auroc'] = auroc
            loss_D['crit_state_auprc'] = auprc
        if metric == 'variance_dist':
            # calculate distribution difference in variance
            # add to loss_D, multiple items
            var_d = variance_dist(pred, true)
            for key, val in var_d.items():
                loss_D[key] = val
        # remove empty key 'variange_dist'
        if 'variance_dist' in loss_D:
            del loss_D['variance_dist']

    return loss_D


def variance_dist(pred, true):
    """Calculates the variance (between data points) for a full trajectory

    Args:
        pred (numpy array): _description_
        true (numpy array): _description_
    """
    pred_var = np.diff(pred, axis=0)
    true_var = np.diff(true, axis=0)
    # convert both to torch tensor and use compute_distances
    pred_var = torch.tensor(pred_var).float()
    true_var = torch.tensor(true_var).float()
    names, values = compute_distribution_distances_new(pred_var, true_var)
    var_met_dict = {name: value for name, value in zip(names, values)}
    return var_met_dict


def critical_state_pred(pred, true, cutoff=-0.91, map_idx = 1):
    """calculate the percentage of critical state predicted
    always in format: HR, MAP
        MAP: 60 cutoff

    Args:
        pred (_type_): _description_
        true (_type_): _description_
        cutoff (float, optional): _description_. Defaults to -0.91.
            -0.91 for normalized (not scaled)
    """
    # return the percentage of critical state predicted
    critical_cutoff = cutoff
    pred_critical = (pred[:, map_idx] < critical_cutoff).astype(int)
    true_critical = (true[:, map_idx] < critical_cutoff).astype(int)
    auroc = roc_auc_score(true_critical, pred_critical)
    auprc = average_precision_score(true_critical, pred_critical)
    return auroc, auprc
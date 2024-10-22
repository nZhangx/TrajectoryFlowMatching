"""

Neural ODE and SDE baseline models.
lr = 1e-3 

"""


import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdiffeq import odeint
from utils.metric_calc import *
class ODEFunc(nn.Module):
    def __init__(self, dim, w=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, dim),
        )

    def forward(self, t, x):
        return self.net(x)

class ODEBaseline(pl.LightningModule):
    def __init__(self, 
                 dim=2, 
                 w=64, 
                 lr=1e-5, 
                 loss_fn=nn.MSELoss(),
                 metrics = ['mse_loss', 'l1_loss']):
        super().__init__()
        self.ode_func = ODEFunc(dim, w)
        self.lr = lr
        self.loss_fn = loss_fn
        self.naming = 'ODEBaseline'
        self.metrics = metrics

    def forward(self, x0, t_span):
        return odeint(self.ode_func, x0, t_span)

    def training_step(self, batch, batch_idx):
        """x0, x0_class, x1, x0_time, x1_time """
        x0, x0_class, x1, x0_time, x1_time = batch
        t_span = x1_time.squeeze() #- x0_time
        print("training_step")
        # print(x0.shape, x1.shape, t_span.shape) # torch.Size([256, 2]) torch.Size([256, 2]) torch.Size([256])
        x_pred = self.forward(x0, t_span)
        loss = self.loss_fn(x_pred[-1], x1.squeeze())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def validation_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time = batch
        t_span = x1_time.squeeze()
        x_pred = self.forward(x0, t_span)
        loss = self.loss_fn(x_pred[-1], x1.squeeze())

        # metrics
        metricsD = metrics_calculation(x_pred[-1], x1, metrics=self.metrics)
        for k, v in metricsD.items():
            self.log(f'{k}_val', v)

        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time = batch
        t_span = x1_time.squeeze()
        x_pred = self.forward(x0, t_span)
        loss = self.loss_fn(x_pred[-1], x1.squeeze())
        # metrics
        metricsD = metrics_calculation(x_pred[-1], x1, metrics=self.metrics)
        for k, v in metricsD.items():
            self.log(f'{k}_test', v)

        self.log('test_loss', loss)
        return loss

# sde
import torchsde

class SDEFunc(nn.Module):
    noise_type = 'diagonal' # diagonal = noise is uncorrelated across dimensions
    sde_type = 'ito'
    def __init__(self, dim, w=64):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(dim, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, dim),
        )
        self.sigma = nn.Sequential(
            nn.Linear(dim, w),
            nn.Tanh(),
            nn.Linear(w, w),
            nn.Tanh(),
            nn.Linear(w, dim),
        )

    def f(self, t, x):
        return self.mu(x)

    def g(self, t, x):
        return self.sigma(x)

class SDEBaseline(pl.LightningModule):
    def __init__(self, 
                 dim=2,
                 w=64, 
                 lr=1e-5, 
                 loss_fn=nn.MSELoss(),
                 metrics = ['mse_loss', 'l1_loss']):
        super().__init__()
        self.sde_func = SDEFunc(dim, w)
        self.lr = lr
        self.loss_fn = loss_fn
        self.naming = 'SDEBaseline'
        self.metrics = metrics

    def forward(self, x0, t_span):
        # print(x0.shape)
        x0 = x0.squeeze()
        batch_size, dim = x0.shape
        bm = torchsde.BrownianInterval(t0=t_span[0], 
                                       t1=t_span[-1], 
                                       dtype=x0.dtype, 
                                       device=x0.device,
                                       size=(batch_size, dim),
                                       levy_area_approximation="space-time")
        return torchsde.sdeint(self.sde_func, x0, t_span, bm=bm)

    def training_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time = batch
        t_span = x1_time.squeeze() #- x0_time
        # x0, x1, t_span = batch
        x_pred = self.forward(x0, t_span)
        loss = self.loss_fn(x_pred[-1], x1.squeeze())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time = batch
        t_span = x1_time.squeeze()
        x_pred = self.forward(x0, t_span)
        loss = self.loss_fn(x_pred[-1], x1.squeeze())
        # metrics
        metricsD = metrics_calculation(x_pred[-1], x1, metrics=self.metrics)
        for k, v in metricsD.items():
            self.log(f'{k}_val', v)
            
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time = batch
        t_span = x1_time.squeeze()
        x_pred = self.forward(x0, t_span)
        loss = self.loss_fn(x_pred[-1], x1.squeeze())
        # metrics
        metricsD = metrics_calculation(x_pred[-1], x1, metrics=self.metrics)
        for k, v in metricsD.items():
            self.log(f'{k}_test', v)
            
        self.log('test_loss', loss)
        return loss
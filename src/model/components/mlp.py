import torch
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torchdyn
from torchdyn.core import NeuralODE
import pytorch_lightning as pl
from torch import optim
import torch.functional as F
import wandb

from utils.visualize import *
from utils.metric_calc import *
from utils.sde import SDE

from model.components.positional_encoding import *
from model.components.grad_util import torch_wrapper_tv
from utils.loss import mse_loss


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, *args, **kwargs):
        return self.net(x)


# conditional liver model
class MLP_conditional_liver(torch.nn.Module):
    """ Conditional with many available classes

    return the class as is
    """
    def __init__(self, dim, treatment_cond, out_dim=None, w=64, time_varying=False, conditional=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            self.out_dim = dim
        self.treatment_cond = treatment_cond
        self.indim = dim + (1 if time_varying else 0) + (self.treatment_cond if conditional else 0)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.indim, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w,self.out_dim),
        )
        self.default_class = 0
        

    def forward(self, x):
        """forward pass
        Assume first two dimensions are x, c, then t
        """
        result = self.net(x)
        # print(result.shape)
        # print(x[:,2:-2].shape)
        return torch.cat([result, x[:,2:-1]], dim=1)
    
class MLP_conditional_liver_pe(torch.nn.Module):
    """ Conditional with many available classes

    return the class as is
    """
    def __init__(self, dim, treatment_cond, out_dim=None, w=64, time_varying=False, conditional=False, time_dim = NUM_FREQS * 2, clip = None):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            self.out_dim = dim 
        self.out_dim += 1
        self.treatment_cond = treatment_cond
        self.dim = dim
        self.indim = dim + (time_dim if time_varying else 0) + (self.treatment_cond if conditional else 0)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.indim, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w,self.out_dim),
        )
        self.default_class = 0
        self.clip = clip

    def encoding_function(self, time_tensor):
        return positional_encoding_tensor(time_tensor)    

    def forward_train(self, x):
        """forward pass
        Assume first two dimensions are x, c, then t
        """
        time_tensor = x[:,-1]
        encoded_time_span = self.encoding_function(time_tensor).reshape(-1, NUM_FREQS * 2)
        new_x = torch.cat([x[:,:-1], encoded_time_span], dim = 1).to(torch.float32)
        result = self.net(new_x)
        return torch.cat([result[:,:-1], x[:,self.dim:-1],result[:,-1].unsqueeze(1)], dim=1)

    def forward(self, x):
        """ call forward_train for training
            x here is x_t
            xt = (t)x1 + (1-t)x0
            (xt - tx1)/(1-t) = x0
        """
        x1 = self.forward_train(x)
        x1_coord = x1[:,:self.dim]
        t = x[:,-1]
        pred_time_till_t1 = x1[:,-1]
        x_coord = x[:,:self.dim]
        if self.clip is None:
            vt = (x1_coord - x_coord)/(pred_time_till_t1)
        else:
            vt = (x1_coord - x_coord)/torch.clip((pred_time_till_t1),min=self.clip)
        final_vt = torch.cat([vt, torch.zeros_like(x[:,self.dim:-1])], dim=1)
        return final_vt

class MLP_Cond_Module(pl.LightningModule):
    def __init__(self, 
                 treatment_cond,
                 dim=2, 
                 w=64, 
                 time_varying=True, 
                 conditional=True,
                 lr=1e-6,
                 sigma = 0.1, 
                 loss_fn = mse_loss,
                 metrics = ['mse_loss', 'l1_loss'],
                 implementation = "ODE", # can be SDE
                 sde_noise = 0.1,
                 clip = None, # float
                 naming = None,
                 ):
        super().__init__()
        self.model = MLP_conditional_liver_pe(dim=dim, 
                                              w=w, 
                                              time_varying=time_varying, 
                                              conditional=conditional, 
                                              treatment_cond=treatment_cond,
                                              clip = clip)
        self.loss_fn = loss_fn
        self.save_hyperparameters()
        self.dim = dim
        # self.out_dim = out_dim
        self.w = w
        self.time_varying = time_varying
        self.conditional = conditional
        self.treatment_cond = treatment_cond
        self.lr = lr
        self.sigma = sigma
        self.naming = "MLP_Cond_Module_"+implementation if naming is None else naming
        self.metrics = metrics
        self.implementation = implementation
        self.sde_noise = sde_noise
        self.clip = clip
        
            
    def __convert_tensor__(self, tensor):
        return tensor.to(torch.float32)

    def __x_processing__(self, x0, x1, t0, t1):
        # squeeze xs (prevent mismatch)
        x0 = x0.squeeze(0)
        x1 = x1.squeeze(0)
        t0 = t0.squeeze()
        t1 = t1.squeeze()

        t = torch.rand(x0.shape[0],1).to(x0.device)
        mu_t = x0 * (1 - t) + x1 * t
        data_t_diff = (t1 - t0).unsqueeze(1)
        x = mu_t + self.sigma * torch.randn(x0.shape[0], self.dim).to(x0.device)
        ut = (x1 - x0) / (data_t_diff + 1e-4)
        t_model = t * data_t_diff + t0.unsqueeze(1)
        futuretime = t1 - t_model
        return x, ut, t_model, futuretime, t
    
    def training_step(self, batch, batch_idx):
        """_summary_

        Args:
            batch (list of output): x0_values, x0_classes, x1_values, times_x0, times_x1
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        x0, x0_class, x1, x0_time, x1_time = batch
        x0, x0_class, x1, x0_time, x1_time = self.__convert_tensor__(x0), self.__convert_tensor__(x0_class), self.__convert_tensor__(x1), self.__convert_tensor__(x0_time), self.__convert_tensor__(x1_time)


        x, ut, t_model, futuretime, t = self.__x_processing__(x0, x1, x0_time, x1_time)
        

        if len(x0_class.shape) == 3:
            x0_class = x0_class.squeeze(0)

        if self.conditional:
            in_tensor = torch.cat([x,x0_class, t_model], dim = -1)
        else:
            in_tensor = torch.cat([x, t_model], dim = -1)
        xt = self.model.forward_train(in_tensor)

        if self.implementation == "SDE":
            variance = t*(1-t)*(self.sde_noise ** 2)
            noise = torch.randn_like(xt[:,:self.dim]) * torch.sqrt(variance)
            loss = self.loss_fn(xt[:,:self.dim]+noise, x1) + self.loss_fn(xt[:,-1], futuretime)
        else:
            loss = self.loss_fn(xt[:,:self.dim], x1) + self.loss_fn(xt[:,-1], futuretime)
        self.log('train_loss', loss)
        return loss

    def config_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def validation_step(self, batch, batch_idx):
        """validation_step

        Args:
            batch (_type_): batch size of 1 (since uneven)
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        loss, pairs, metricD = self.test_func_step(batch, batch_idx, mode='val')
        self.log('val_loss', loss)
        for key, value in metricD.items():
            self.log(key+"_val", value)
        # return total_loss, traj_pairs
        return {'val_loss':loss, 'traj_pairs':pairs}

    def test_step(self, batch, batch_idx):
        loss, pairs, metricD = self.test_func_step(batch, batch_idx, mode='test')
        self.log('test_loss', loss)
        for key, value in metricD.items():
            self.log(key+"_test", value)
        return {'test_loss':loss, 'traj_pairs':pairs}
    
    def test_func_step(self, batch, batch_idx, mode='none'):
        """assuming each is one patient/batch"""
        total_loss = []
        traj_pairs = []

        x0_values, x0_classes, x1_values, times_x0, times_x1 = batch
        times_x0 = times_x0.squeeze()
        times_x1 = times_x1.squeeze()

        # print(x0_values.shape)
        # print(x1_values.shape)
        full_traj = torch.cat([x0_values[0,0,:self.dim].unsqueeze(0), 
                               x1_values[0,:,:self.dim]], 
                               dim=0)
        full_time = torch.cat([times_x0[0].unsqueeze(0), times_x1], dim=0)
        ind_loss, pred_traj = self.test_trajectory(batch)
        total_loss.append(ind_loss)
        traj_pairs.append([full_traj, pred_traj])

        full_traj = full_traj.detach().cpu().numpy()
        pred_traj = pred_traj.detach().cpu().numpy()
        full_time = full_time.detach().cpu().numpy()

        # graph
        fig = plot_3d_path_ind(pred_traj, 
                               full_traj, 
                               t_span=full_time,
                               title="{}_trajectory_patient_{}".format(mode, batch_idx))
        if self.logger:
            # may cause problem if wandb disabled
            self.logger.experiment.log({"{}_trajectory_patient_{}".format(mode, batch_idx): wandb.Image(fig)})
        
        plt.close(fig)

        # metrics
        metricD = metrics_calculation(pred_traj, full_traj, metrics=self.metrics)
        return np.mean(total_loss), traj_pairs, metricD

    def test_trajectory(self,pt_tensor):
        if self.implementation == "ODE":
            return self.test_trajectory_ode(pt_tensor)
        elif self.implementation == "SDE":
            return self.test_trajectory_sde(pt_tensor)
    
    def test_trajectory_ode(self,pt_tensor):
        """test_trajectory

        Args:
            pt_tensor (numpy.array): (x0_values, x0_classes, x1_values, times_x0, times_x1),


        Returns:
            mse_all, total_pred_tensor: _description_
        """
        node = NeuralODE(
            torch_wrapper_tv(self.model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        total_pred = []
        mse = []
        t_max = 0

        x0_values, x0_classes, x1_values, times_x0, times_x1 = pt_tensor
        # squeeze all
        x0_values = x0_values.squeeze(0)
        x1_values = x1_values.squeeze(0)
        times_x0 = times_x0.squeeze()
        times_x1 = times_x1.squeeze()
        x0_classes = x0_classes.squeeze()

        if len(x0_classes.shape) == 1:
            x0_classes = x0_classes.unsqueeze(1)



        total_pred.append(x0_values[0].unsqueeze(0))
        len_path = x0_values.shape[0]
        assert len_path == x1_values.shape[0]
        for i in range(len_path): 
            t_max = (times_x1[i]-times_x0[i]) # calculate time difference (cumulative)
            time_span = self.__convert_tensor__(torch.linspace(times_x0[i], times_x1[i], 10)).to(x0_values.device)
            with torch.no_grad():
                # get last pred, if none then use startpt
                if i == 0:
                    if self.conditional:
                        testpt = torch.cat([x0_values[i].unsqueeze(0),x0_classes[i].unsqueeze(0)],dim=1)
                    else:
                        testpt = x0_values[i].unsqueeze(0)
                else: # incorporate last prediction
                    if self.conditional:
                        testpt = torch.cat([pred_traj, x0_classes[i].unsqueeze(0)], dim=1)
                    else:
                        testpt = pred_traj
                traj = node.trajectory(
                    testpt,
                    t_span=time_span, 
                )
            pred_traj = traj[-1,:,:self.dim]
            total_pred.append(pred_traj)
            ground_truth_coords = x1_values[i]
            mse.append(self.loss_fn(pred_traj, ground_truth_coords).detach().cpu().numpy())
        mse_all = np.mean(mse)
        total_pred_tensor = torch.stack(total_pred).squeeze(1)
        return mse_all, total_pred_tensor
    
    def configure_optimizers(self):
        # Define the optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def test_trajectory_sde(self,pt_tensor):
        """test_trajectory

        Args:
            pt_tensor (numpy.array): (x0_values, x0_classes, x1_values, times_x0, times_x1),


        Returns:
            mse_all, total_pred_tensor: _description_
        """
        sde = SDE(self.model, noise=0.1)
        total_pred = []
        mse = []
        t_max = 0

        x0_values, x0_classes, x1_values, times_x0, times_x1 = pt_tensor
        # squeeze all
        x0_values = x0_values.squeeze(0)
        x1_values = x1_values.squeeze(0)
        times_x0 = times_x0.squeeze()
        times_x1 = times_x1.squeeze()
        x0_classes = x0_classes.squeeze()

        if len(x0_classes.shape) == 1:
            x0_classes = x0_classes.unsqueeze(1)



        total_pred.append(x0_values[0].unsqueeze(0))
        len_path = x0_values.shape[0]
        assert len_path == x1_values.shape[0]
        for i in range(len_path): 
            t_max = (times_x1[i]-times_x0[i]) 
            time_span = self.__convert_tensor__(torch.linspace(times_x0[i], times_x1[i], 10)).to(x0_values.device)
            
            with torch.no_grad():
                # get last pred, if none then use startpt
                if i == 0:
                    if self.conditional:
                        testpt = torch.cat([x0_values[i].unsqueeze(0),x0_classes[i].unsqueeze(0)],dim=1)
                    else:
                        testpt = x0_values[i].unsqueeze(0)
                else: # incorporate last prediction
                    if self.conditional:
                        testpt = torch.cat([pred_traj, x0_classes[i].unsqueeze(0)], dim=1)
                    else:
                        testpt = pred_traj
                traj = self._sde_solver(sde, testpt, time_span)

            pred_traj = traj[-1,:,:self.dim]
            total_pred.append(pred_traj)
            ground_truth_coords = x1_values[i]
            mse.append(self.loss_fn(pred_traj, ground_truth_coords).detach().cpu().numpy())
        mse_all = np.mean(mse)
        total_pred_tensor = torch.stack(total_pred).squeeze(1)
        return mse_all, total_pred_tensor


    def _sde_solver(self, sde, initial_state, time_span):
        dt = time_span[1] - time_span[0]  # Time step
        current_state = initial_state
        trajectory = [current_state]

        for t in time_span[1:]:
            drift = sde.f(t, current_state)
            diffusion = sde.g(t, current_state)
            noise = torch.randn_like(current_state) * torch.sqrt(dt)
            current_state = current_state + drift * dt + diffusion * noise
            trajectory.append(current_state)

        return torch.stack(trajectory)
        



class MLP_conditional_liver_pe_memory(torch.nn.Module):
    """ Conditional with many available classes

    return the class as is
    """
    def __init__(self, 
                 dim, 
                 treatment_cond,
                 memory, # how many time steps
                 out_dim=None, 
                 w=64, 
                 time_varying=False, 
                 conditional=False,  
                 time_dim = NUM_FREQS * 2,
                 clip = None,
                 ):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            self.out_dim = dim 
        self.out_dim += 1 # for the time dimension
        self.treatment_cond = treatment_cond
        self.memory = memory
        self.dim = dim
        self.indim = dim + (time_dim if time_varying else 0) + (self.treatment_cond if conditional else 0) + (dim * memory)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.indim, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w,self.out_dim),
        )
        self.default_class = 0
        self.clip = clip

    def encoding_function(self, time_tensor):
        return positional_encoding_tensor(time_tensor)    

    def forward_train(self, x):
        """forward pass
        Assume first two dimensions are x, c, then t
        """
        time_tensor = x[:,-1]
        encoded_time_span = self.encoding_function(time_tensor).reshape(-1, NUM_FREQS * 2)
        new_x = torch.cat([x[:,:-1], encoded_time_span], dim=1)
        result = self.net(new_x)
        return torch.cat([result[:,:-1], x[:,self.dim:-1], result[:,-1].unsqueeze(1)], dim=1)

    def forward(self, x):
        """ call forward_train for training
            x here is x_t
            xt = (t)x1 + (1-t)x0
            (xt - tx1)/(1-t) = x0
        """
        x1 = self.forward_train(x)
        x1_coord = x1[:,:self.dim]
        t = x[:,-1]
        pred_time_till_t1 = x1[:,-1]
        x_coord = x[:,:self.dim]
        if self.clip is None:
            vt = (x1_coord - x_coord)/(pred_time_till_t1)
        else:
            vt = (x1_coord - x_coord)/torch.clip((pred_time_till_t1),min=self.clip)

        final_vt = torch.cat([vt, torch.zeros_like(x[:,self.dim:-1])], dim=1)
        return final_vt
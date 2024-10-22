import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdiffeq import odeint
from utils.metric_calc import *
from utils.latent_ode_utils import *

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.Tanh(),
            nn.Linear(20, latent_dim),
        )

    def forward(self, x):
        return self.net(x)
    


class Encoder_z0_ODE_RNN(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, latent_dim, input_dim, z0_diffeq_solver = None, 
        z0_dim = None, GRU_update = None, 
        n_gru_units = 100, 
        ):
        
        super(Encoder_z0_ODE_RNN, self).__init__()

        self.z0_dim = latent_dim
        self.GRU_update = GRU_unit(latent_dim, input_dim, 
            n_units = n_gru_units)
            # device=device).to(device)

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        # self.device = device
        self.extra_info = None

        self.transform_z0 = nn.Sequential(
           nn.Linear(latent_dim * 2, 100),
           nn.Tanh(),
           nn.Linear(100, self.z0_dim * 2),)


    def forward(self, x1):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        # xi is timepoint i of the data
        device_of_x1 = x1.device
        x1_len = x1.shape[1]

        prev_y = torch.zeros((1, x1_len, self.latent_dim)).to(device_of_x1)
        prev_std = torch.zeros((1, x1_len, self.latent_dim)).to(device_of_x1)

        # x1 = x1.unsqueeze(0)
        
        last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, x1)
        
        means_z0 = last_yi.reshape(1, x1_len, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, x1_len, self.latent_dim)

        mean_z0, std_z0 = split_last_dim(self.transform_z0( torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()

        return mean_z0, std_z0



from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent
from model.base_models import VAE_Baseline


class VAE(nn.Module):
    def __init__(self, encoder, decoder, ode_func, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ode_func = ode_func
        self.latent_dim = latent_dim

    def forward(self, x, t, reverse=False):
        """Forward pass for training with the option to reverse time.
        
        Args:
            x: Input data (e.g., sequence of observations).
            t: Corresponding time points for the input data.
            reverse: Whether to reverse time for the ODE solver.
        """
        if len(t.shape)>1:
            t = t.squeeze()

        if reverse:
            # Training mode: reverse time
            t = torch.flip(t, dims=[0])
            x = torch.flip(x, dims=[1])

        mu, std_z0 = self.encoder(x)
        z = self.reparameterize(mu, std_z0)
        z_pred = odeint(self.ode_func, z, t)
        x_pred = self.decoder(z_pred)

        if reverse:
            # Flip predictions back to original time order
            x_pred = torch.flip(x_pred, dims=[1])

        return x_pred, mu, std_z0

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    
    def extrapolate(self,z0, t):
        """predict future states - call during val/test
        
        Args:
            z0: Initial latent state from which to extrapolate.
            t: Future time points to predict.
        """
        # do one at a time
        z_pred = odeint(self.ode_func, z0, t)
        return self.decoder(z_pred)

    

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, latent_dim),
        )

    def forward(self, t, x):
        return self.net(x)

class LatentODE_pl(pl.LightningModule):
    def __init__(self,
            input_dim,
            latent_dim, 
            output_dim,
            lr=1e-3,
            loss_fn=nn.MSELoss(),
            metrics = ['mse_loss', 'l1_loss'],
        ):
        super().__init__()
        self.encoder = Encoder_z0_ODE_RNN(latent_dim,input_dim)
        self.decoder = Decoder(latent_dim, output_dim)
        self.vae = VAE(self.encoder, self.decoder, LatentODEFunc(latent_dim), latent_dim)
        self.lr = lr
        self.loss_fn = loss_fn
        self.naming = 'LatentODE_RNN'
        self.metrics = metrics

    def forward(self, x, x_time, mode='train'):
        if mode == 'train':
            reverse = True
        else:
            reverse = False
        x_pred, mu, std_z0 = self.vae(x, x_time, reverse=reverse)
        return x_pred, mu, std_z0

    def training_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time  = batch
        t_span = x1_time.squeeze()
        x_pred, mu, std_z0 = self.forward(x1, x1_time, mode='train')
        # loss = self.loss_fn(x_pred[-1], x1)
        loss = self.loss_function(x_pred[-1], x0, mu, std_z0)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time  = batch
        x_pred, x1 = self.testing_vae(batch)
        loss = self.loss_fn(x_pred[-1], x1)

        # metrics
        metricD = metrics_calculation(x_pred[-1], x1, metrics=self.metrics)
        for key, value in metricD.items():
            self.log(key+"_val", value)

        loss = self.loss_fn(x_pred[-1], x1)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time  = batch
        x_pred, x1 = self.testing_vae(batch)
        loss = self.loss_fn(x_pred[-1], x1)

        # Calculate metrics
        metricD = metrics_calculation(x_pred[-1], x1, metrics=self.metrics)

        for key, value in metricD.items():
            self.log(key+"_test", value)

        self.log('test_loss', loss)
        return loss



    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss


    def sample(self, num_samples=1):
        """Sample from the latent space.
        
        Args:
            num_samples: Number of samples to generate.
        """
        z = torch.randn(num_samples, self.latent_dim)
        return self.decoder(z)
    
    def testing_vae(self, batch):
        x0, x0_class, x1, x0_time, x1_time  = batch
        t_span = x1_time.squeeze()
        z0_mean, z0_logvar = self.encoder(x0)
        z0 = self.vae.reparameterize(z0_mean, z0_logvar) # sample
        #@HERE: squeeze z0?
        x_pred = self.vae.extrapolate(z0, t_span)
        return x_pred, x1


class LatentODE_deprecated(pl.LightningModule):
    def __init__(self, 
                 input_dim, 
                 latent_dim, 
                 output_dim, 
                 lr=1e-3, 
                 loss_fn=nn.MSELoss(),
                 metrics = ['mse_loss', 'l1_loss']):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)
        self.ode_func = LatentODEFunc(latent_dim)
        self.lr = lr
        self.loss_fn = loss_fn
        self.naming = 'LatentODE'
        self.metrics = metrics

    def forward(self, x0, t_span):
        z0 = self.encoder(x0)
        z_pred = odeint(self.ode_func, z0, t_span)
        x_pred = self.decoder(z_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time  = batch
        t_span = x1_time.squeeze()
        x_pred = self.forward(x0, t_span)
        loss = self.loss_fn(x_pred[-1], x1)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time  = batch
        t_span = x1_time.squeeze()
        x_pred = self.forward(x0, t_span)

        # metrics
        metricD = metrics_calculation(x_pred[-1], x1, metrics=self.metrics)
        for key, value in metricD.items():
            self.log(key+"_val", value)

        loss = self.loss_fn(x_pred[-1], x1)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time  = batch
        t_span = x1_time.squeeze()
        x_pred = self.forward(x0, t_span)
        loss = self.loss_fn(x_pred[-1], x1)

        # Calculate metrics
        metricD = metrics_calculation(x_pred[-1], x1, metrics=self.metrics)

        for key, value in metricD.items():
            self.log(key+"_test", value)

        self.log('test_loss', loss)
        return loss
 
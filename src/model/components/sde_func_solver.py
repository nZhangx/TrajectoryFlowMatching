import torch

class SDE_func_solver(torch.nn.Module):

    noise_type = "diagonal"
    sde_type = "ito"

    # noise is sigma in this notebook for the equation sigma * (t * (1 - t))
    def __init__(self, ode_drift, noise, reverse=False):
        super().__init__()
        self.drift = ode_drift
        self.reverse = reverse
        self.noise = noise # changeable, a model itself

    # Drift
    def f(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        return self.drift(x)

    # Diffusion
    def g(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        noise_result = self.noise(x)
        return noise_result* torch.sqrt(t * (1 - t))
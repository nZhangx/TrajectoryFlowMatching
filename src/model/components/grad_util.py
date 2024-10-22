import torch

class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        return self.model(x)
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
    
class torch_wrapper_tv(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        # print(x.shape, t.shape)
        # print(t)
        # return self.model(x)
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))



class torch_wrapper_cond(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        # unpack the input
        # class_cond = torch.zeros(x.shape[0],1)
        input = torch.cat([x, t.repeat(x.shape[0])[:, None]], 1)
        print(input.shape)
        return self.model(input)
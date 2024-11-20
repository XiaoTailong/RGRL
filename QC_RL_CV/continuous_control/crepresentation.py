import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
from tqdm import tqdm
import random

class RepresentationNetwork(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim):
        """
        A neural network with skip connections which converts the input observable and
        its corresponding expectation into a representation which is more dense than the input.
        """
        super(RepresentationNetwork, self).__init__()
        # Final representation size
        self.r_dim = k = r_dim
        self.linear1 = nn.Linear(x_dim, k)
        self.linear2 = nn.Linear(k + v_dim, k)
        # self.linear3 = nn.Linear(k + v_dim + x_dim, k)

    def forward(self, x, v):
        x = F.relu(self.linear1(x))
#         print(x.shape)
        merge = torch.cat([x, v], dim=1)
        r = F.relu(self.linear2(merge))
        #         merge = torch.cat([r,x,v],dim=1)
        #         r = F.tanh(self.linear3(merge))
        return r

class RepresentationNetwork2(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim):
        """
        A neural network with skip connections which converts the input observable and
        its corresponding expectation into a representation which is more dense than the input.
        """
        super(RepresentationNetwork2, self).__init__()
        # Final representation size
        self.r_dim = k = r_dim
        self.linear1 = nn.Linear(v_dim, k)
        self.linear2 = nn.Linear(x_dim, k)
        self.linear3 = nn.Linear(2*k, k)
        self.linear4 = nn.Linear(k , k)

    def forward(self, x, v):
        v = F.relu(self.linear1(v))
        x = F.relu(self.linear2(x))
        merge = torch.cat([x, v], dim=1)
        r = F.relu(self.linear3(merge))
        r = F.relu(self.linear4(r))
        return r
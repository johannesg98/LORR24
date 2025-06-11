from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.data import Data, Batch
import numpy as np
import os

class GNNActor(nn.Module):
    def __init__(self, in_channels, hidden_size=32, act_dim=6, edge_limit=0.5, out_channel_fac=2, edge_feature_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channel_fac * in_channels
        self.act_dim = act_dim
        self.conv1 = TransformerConv(in_channels, self.out_channels, edge_dim=edge_feature_dim)
        self.lin1 = nn.Linear(in_channels+self.out_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 2)
        self.lin1_norm = nn.LayerNorm(hidden_size)
        self.lin2_norm = nn.LayerNorm(hidden_size)
        


    def forward(self, state, edge_index, edge_attr, deterministic=False, return_dist=False, return_raw=False):
        
        out1 = F.relu(self.conv1(state, edge_index, edge_attr=edge_attr))

        state = state.reshape(-1, self.act_dim, self.in_channels)
        out1 = out1.reshape(-1, self.act_dim, self.out_channels)
        
        x = torch.cat((out1, state), dim=-1)
        
        x = F.leaky_relu(self.lin1_norm(self.lin1(x)))
        x = F.leaky_relu(self.lin2_norm(self.lin2(x)))
        x = torch.sigmoid(self.lin3(x))
        continous_action = x
        
        
    
        if return_dist:
            return torch.distributions.Bernoulli(probs=continous_action)
        if return_raw:
            return continous_action, None
        if deterministic:
            action = (continous_action > 0.5).int()  # Convert to binary action
            return action, None
        else:
            eps = 0.3
            continous_action = (1-eps) * continous_action + eps * torch.rand_like(continous_action)
            dist = torch.distributions.Bernoulli(probs=continous_action)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_prob = log_prob.sum(dim=-1).sum(dim=-1)
            return action.int(), log_prob
    



class GNNCritic(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6, edge_limit=0.5, out_channel_fac=2, edge_feature_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channel_fac * in_channels
        self.act_dim = act_dim
        self.conv1 = TransformerConv(in_channels, self.out_channels, edge_dim=edge_feature_dim)
        self.lin1 = nn.Linear(in_channels+self.out_channels+2, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.lin1_norm = nn.LayerNorm(hidden_size)
        self.lin2_norm = nn.LayerNorm(hidden_size)

    def forward(self, state, edge_index, edge_attr, action):
        
        out1 = F.relu(self.conv1(state, edge_index, edge_attr=edge_attr))

        state = state.reshape(-1, self.act_dim, self.in_channels)
        out1 = out1.reshape(-1, self.act_dim, self.out_channels)
        action = action.reshape(-1, self.act_dim, 2)


        concat = torch.cat((out1, state, action), dim=-1) # (B, N, C)


        x = F.relu(self.lin1_norm(self.lin1(concat)))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x
    

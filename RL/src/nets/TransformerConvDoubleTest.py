from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data, Batch
import numpy as np
import os

class GNNActor(nn.Module):
    def __init__(self, in_channels, hidden_size=32, act_dim=6, edge_limit=0.5, out_channel_fac=2, edge_feature_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channel_fac * in_channels
        self.conv2_in_channels = 10
        self.conv2_out_channels = 20
        self.act_dim = act_dim
        self.conv1 = TransformerConv(in_channels, self.out_channels, edge_dim=edge_feature_dim)
        self.lin1 = nn.Linear(in_channels+self.out_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, self.conv2_in_channels)
        self.conv2 = TransformerConv(self.conv2_in_channels, self.conv2_out_channels, edge_dim=edge_feature_dim)
        self.lin4 = nn.Linear(self.conv2_in_channels + self.conv2_out_channels, self.conv2_in_channels)
        self.conv3 = TransformerConv(self.conv2_in_channels, self.conv2_out_channels, edge_dim=edge_feature_dim)
        self.lin5 = nn.Linear(self.conv2_in_channels + self.conv2_out_channels, 1)


        self.lin1_norm = nn.LayerNorm(hidden_size)
        self.lin2_norm = nn.LayerNorm(hidden_size)
        self.lin3_norm = nn.LayerNorm(self.conv2_in_channels)
        self.conv2_norm = nn.LayerNorm(self.conv2_out_channels)
        self.lin4_norm = nn.LayerNorm(self.conv2_in_channels)
        self.conv3_norm = nn.LayerNorm(self.conv2_out_channels)

    def forward(self, state, edge_index, edge_attr, deterministic=False, return_dist=False, return_raw=False):
        
        out = F.relu(self.conv1(state, edge_index, edge_attr=edge_attr))

        state = state.reshape(-1, self.act_dim, self.in_channels)
        out = out.reshape(-1, self.act_dim, self.out_channels)

        x = torch.cat((out, state), dim=-1)
        
        x = F.leaky_relu(self.lin1_norm(self.lin1(x)))
        x = F.leaky_relu(self.lin2_norm(self.lin2(x)))
        x = F.leaky_relu(self.lin3_norm(self.lin3(x)))


        data_list = []
        for i in range(x.shape[0]):
            data_list.append(Data(x=x[i]))
        batch = Batch.from_data_list(data_list)


        out = F.relu(self.conv2_norm(self.conv2(batch.x, edge_index, edge_attr=edge_attr)))

        out = out.reshape(-1, self.act_dim, self.conv2_out_channels)
        x = torch.cat((out, x), dim=-1)

        x = F.leaky_relu(self.lin4_norm(self.lin4(x)))


        data_list = []
        for i in range(x.shape[0]):
            data_list.append(Data(x=x[i]))
        batch = Batch.from_data_list(data_list)

        out = F.relu(self.conv3_norm(self.conv3(batch.x, edge_index, edge_attr=edge_attr)))

        out = out.reshape(-1, self.act_dim, self.conv2_out_channels)

        x = torch.cat((out, x), dim=-1)
        x = F.softplus(self.lin5(x))



        









        concentration = x.squeeze(-1)
        if return_dist:
            return Dirichlet(concentration + 1e-20)
        if return_raw:
            action = concentration
            log_prob = None
        elif deterministic:
            action = concentration / (concentration.sum(dim=-1, keepdim=True) + 1e-20)  # Normalize
            log_prob = None
        else:
            m = Dirichlet(concentration + 1e-20)
            action = m.rsample()
            log_prob = m.log_prob(action)
        return action, log_prob
    



class GNNCritic(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6, edge_limit=0.5, out_channel_fac=2, edge_feature_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channel_fac * in_channels
        self.conv2_in_channels = 10
        self.conv2_out_channels = 20
        self.act_dim = act_dim
        self.conv1 = TransformerConv(in_channels, self.out_channels, edge_dim=edge_feature_dim)
        self.lin1 = nn.Linear(in_channels+self.out_channels+1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, self.conv2_in_channels)
        self.conv2 = TransformerConv(self.conv2_in_channels, self.conv2_out_channels, edge_dim=edge_feature_dim)
        self.lin4 = nn.Linear(self.conv2_in_channels + self.conv2_out_channels, self.conv2_in_channels)
        self.conv3 = TransformerConv(self.conv2_in_channels, self.conv2_out_channels, edge_dim=edge_feature_dim)
        self.lin5 = nn.Linear(self.conv2_in_channels + self.conv2_out_channels, self.conv2_in_channels + self.conv2_out_channels)
        self.lin6 = nn.Linear(self.conv2_in_channels + self.conv2_out_channels, 1)

        self.lin1_norm = nn.LayerNorm(hidden_size)
        self.lin2_norm = nn.LayerNorm(hidden_size)
        self.lin3_norm = nn.LayerNorm(self.conv2_in_channels)
        self.conv2_norm = nn.LayerNorm(self.conv2_out_channels)
        self.lin4_norm = nn.LayerNorm(self.conv2_in_channels)
        self.conv3_norm = nn.LayerNorm(self.conv2_out_channels)
        self.lin5_norm = nn.LayerNorm(self.conv2_in_channels + self.conv2_out_channels)
        
    def forward(self, state, edge_index, edge_attr, action):
        
        out1 = F.relu(self.conv1(state, edge_index, edge_attr=edge_attr))

        action_flattened = action.reshape(-1, 1)
        x = torch.cat((out1, state, action_flattened), dim=-1)

        x = F.leaky_relu(self.lin1_norm(self.lin1(x)))
        x = F.leaky_relu(self.lin2_norm(self.lin2(x)))
        x = F.leaky_relu(self.lin3_norm(self.lin3(x)))

        out = F.relu(self.conv2_norm(self.conv2(x, edge_index, edge_attr=edge_attr)))

        x = torch.cat((out, x), dim=-1)

        x = F.leaky_relu(self.lin4_norm(self.lin4(x)))

        out = F.relu(self.conv3_norm(self.conv3(x, edge_index, edge_attr=edge_attr)))

        x = torch.cat((out, x), dim=-1)

        x = x.reshape(-1, self.act_dim, self.conv2_in_channels + self.conv2_out_channels)

        x = F.leaky_relu(self.lin5_norm(self.lin5(x))) # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin6(x).squeeze(-1)  # (B)
        return x




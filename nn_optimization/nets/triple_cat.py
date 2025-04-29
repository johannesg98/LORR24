from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.nn import GCNConv


class GNNActor(nn.Module):
    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, in_channels)
        self.conv3 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(4*in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, deterministic=False, return_dist=False, return_raw=False):
        out1 = F.relu(self.conv1(state, edge_index))
        out2 = F.relu(self.conv2(out1, edge_index))
        out3 = F.relu(self.conv3(out2, edge_index))
        if torch.isnan(out3).any():
            print("NaN values detected in out!")
        x = torch.cat((out1, out2, out3, state), dim=-1)
        # x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
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
        regularize = concentration.abs().mean()
        return action, log_prob, regularize
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
        self.globLin1 = nn.Linear(in_channels*act_dim, 256)
        self.globLin2 = nn.Linear(256, 10)
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels+10, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, deterministic=False, return_dist=False):
        out = F.relu(self.conv1(state, edge_index))
        if torch.isnan(out).any():
            print("NaN values detected in out!")
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)

        xglob = F.leaky_relu(self.globLin1(state.reshape(-1, self.in_channels*self.act_dim)))
        xglob = F.leaky_relu(self.globLin2(xglob))
        xglob = xglob.unsqueeze(1)
        xglob = xglob.expand(-1, self.act_dim, -1)
        x = torch.cat((x, xglob), dim=-1)

        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
        concentration = x.squeeze(-1)
        if return_dist:
            return Dirichlet(concentration + 1e-20)
        if deterministic:
            action = concentration / (concentration.sum(dim=-1, keepdim=True) + 1e-20)  # Normalize
            log_prob = None
        else:
            m = Dirichlet(concentration + 1e-20)
            action = m.rsample()
            log_prob = m.log_prob(action)
        regularize = concentration.abs().mean()
        return action, log_prob, regularize
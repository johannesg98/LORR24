from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Your GNNActor model definition
class GNNActor(nn.Module):
    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, 10)
        self.conv2 = GCNConv(10, 21)
        self.conv3 = GCNConv(21, 21)
        self.conv4 = GCNConv(in_channels, in_channels)
        self.conv5 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels+1+10+21+21+21+21, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, deterministic=False, return_dist=False, return_raw=False):
        out1 = F.relu(self.conv1(state, edge_index))
        out2 = F.relu(self.conv2(out1, edge_index))
        out3 = F.relu(self.conv3(out2, edge_index))
        out4 = F.relu(self.conv3(out3, edge_index))
        out5 = F.relu(self.conv3(out4, edge_index))
        if torch.isnan(out5).any():
            print("NaN values detected in out!")
        total_agents = state[...,1].sum(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, self.act_dim, -1)
        state = torch.cat((state, total_agents), dim=-1)
        x = torch.cat((out1, out2, out3, out4, out5, state), dim=-1)
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
    

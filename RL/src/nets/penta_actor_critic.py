from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNActorPenta(nn.Module):
    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, in_channels)
        self.conv3 = GCNConv(in_channels, in_channels)
        self.conv4 = GCNConv(in_channels, in_channels)
        self.conv5 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(6*in_channels+1+2, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.pos_feat = self.get_positions()

    def forward(self, state, edge_index, deterministic=False, return_dist=False, return_raw=False):
        out1 = F.relu(self.conv1(state, edge_index))
        out2 = F.relu(self.conv2(out1, edge_index))
        out3 = F.relu(self.conv3(out2, edge_index))
        out4 = F.relu(self.conv3(out3, edge_index))
        out5 = F.relu(self.conv3(out4, edge_index))

        out1 = out1.reshape(-1, self.act_dim, self.in_channels)
        out2 = out2.reshape(-1, self.act_dim, self.in_channels)
        out3 = out3.reshape(-1, self.act_dim, self.in_channels)
        out4 = out4.reshape(-1, self.act_dim, self.in_channels)
        out5 = out5.reshape(-1, self.act_dim, self.in_channels)
        state = state.reshape(-1, self.act_dim, self.in_channels)

        total_agents = state[...,1].sum(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, self.act_dim, -1)
        # if normalized:
        #     total_agents = torch.round(total_agents/agent_scale_fac)
        positions = self.pos_feat.unsqueeze(0).expand(state.shape[0], -1, -1)
        state = torch.cat((state, total_agents, positions), dim=-1)
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
    
    def get_positions(self):
        pos_indices = [120,124,128,132,136,140,144,148,152,237,241,245,249,253,257,261,265,269,354,358,362,366,370,374,378,382,386,471,475,479,483,487,491,495,499,503,588,592,596,600,604,608,612,616,620,705,709,713,717,721,725,729,733,737,822,826,830,834,838,842,846,850,854,48,53,60,67,73,157,352,388,583,586,817,901,906,913,920,926]
        height = 25
        width = 39
        nNodesss = 79
        position_features = torch.zeros((nNodesss,2))
        for i,pos_idx in enumerate(pos_indices):
            x = pos_idx % width
            y = pos_idx // width
            x_norm = x / (width-1)
            y_norm = y / (height-1)
            position_features[i,0] = x_norm
            position_features[i,1] = y_norm
        return position_features.to(device)

    


class GNNCriticPenta(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, in_channels)
        self.conv3 = GCNConv(in_channels, in_channels)
        self.conv4 = GCNConv(in_channels, in_channels)
        self.conv5 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(6 * in_channels + 1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out1 = F.relu(self.conv1(state, edge_index))
        out2 = F.relu(self.conv2(out1, edge_index))
        out3 = F.relu(self.conv3(out2, edge_index))
        out4 = F.relu(self.conv3(out3, edge_index))
        out5 = F.relu(self.conv3(out4, edge_index))
        # x = out + state
        # x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        out1 = out1.reshape(-1, self.act_dim, self.in_channels)
        out2 = out2.reshape(-1, self.act_dim, self.in_channels)
        out3 = out3.reshape(-1, self.act_dim, self.in_channels)
        out4 = out4.reshape(-1, self.act_dim, self.in_channels)
        out5 = out5.reshape(-1, self.act_dim, self.in_channels)
        state = state.reshape(-1, self.act_dim, self.in_channels)
        concat = torch.cat([out1, out2, out3, out4, out5, state, action.unsqueeze(-1)], dim=-1)
        x = F.relu(self.lin1(concat))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x
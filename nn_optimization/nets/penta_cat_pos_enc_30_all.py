from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.nn import GCNConv
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Your GNNActor model definition
class GNNActor(nn.Module):
    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        in_channels += 30
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, in_channels)
        self.conv3 = GCNConv(in_channels, in_channels)
        self.conv4 = GCNConv(in_channels, in_channels)
        self.conv5 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(6*in_channels+1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.pos_feat = self.get_positions()

    def forward(self, state, edge_index, deterministic=False, return_dist=False, return_raw=False):
        positions = self.pos_feat.unsqueeze(0).expand(state.shape[0], -1, -1)
        state = torch.cat((state, positions), dim=-1)
        out1 = F.relu(self.conv1(state, edge_index))
        out2 = F.relu(self.conv2(out1, edge_index))
        out3 = F.relu(self.conv3(out2, edge_index))
        out4 = F.relu(self.conv3(out3, edge_index))
        out5 = F.relu(self.conv3(out4, edge_index))
        if torch.isnan(out5).any():
            print("NaN values detected in out!")
        total_agents = state[...,1].sum(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, self.act_dim, -1)
        # if normalized:
        #     total_agents = torch.round(total_agents/agent_scale_fac)
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
    
    def get_positions(self):
        pos_indices = [120,124,128,132,136,140,144,148,152,237,241,245,249,253,257,261,265,269,354,358,362,366,370,374,378,382,386,471,475,479,483,487,491,495,499,503,588,592,596,600,604,608,612,616,620,705,709,713,717,721,725,729,733,737,822,826,830,834,838,842,846,850,854,48,53,60,67,73,157,352,388,583,586,817,901,906,913,920,926]
        height = 25
        width = 39
        nNodesss = 79
        position_features = torch.zeros((nNodesss,30))
        for i,pos_idx in enumerate(pos_indices):
            x = pos_idx % width
            y = pos_idx // width
            x_norm = x / (width-1)
            y_norm = y / (height-1)
            position_features[i,0] = x_norm
            position_features[i,1] = y_norm
            position_features[i,2] = np.sin(x_norm*2*np.pi)+1
            position_features[i,3] = np.cos(x_norm*2*np.pi)+1
            position_features[i,4] = np.sin(y_norm*2*np.pi)+1
            position_features[i,5] = np.cos(y_norm*2*np.pi)+1
            position_features[i,6] = np.sin(x_norm*5*np.pi)+1
            position_features[i,7] = np.cos(x_norm*5*np.pi)+1
            position_features[i,8] = np.sin(y_norm*5*np.pi)+1
            position_features[i,9] = np.cos(y_norm*5*np.pi)+1
            position_features[i,10] = np.sin(x_norm*12*np.pi)+1
            position_features[i,11] = np.cos(x_norm*12*np.pi)+1
            position_features[i,12] = np.sin(y_norm*12*np.pi)+1
            position_features[i,13] = np.cos(y_norm*12*np.pi)+1
            position_features[i,14] = np.sin(x_norm*30*np.pi)+1
            position_features[i,15] = np.cos(x_norm*30*np.pi)+1
            position_features[i,16] = np.sin(y_norm*30*np.pi)+1
            position_features[i,17] = np.cos(y_norm*30*np.pi)+1
            position_features[i,18] = np.sin(x_norm*100*np.pi)+1
            position_features[i,19] = np.cos(x_norm*100*np.pi)+1
            position_features[i,20] = np.sin(y_norm*100*np.pi)+1
            position_features[i,21] = np.cos(y_norm*100*np.pi)+1
            position_features[i,22] = np.sin(x_norm*100*np.pi)+1
            position_features[i,23] = np.cos(x_norm*100*np.pi)+1
            position_features[i,24] = np.sin(y_norm*100*np.pi)+1
            position_features[i,25] = np.cos(y_norm*100*np.pi)+1
            position_features[i,26] = np.sin(x_norm*100*np.pi)+1
            position_features[i,27] = np.cos(x_norm*100*np.pi)+1
            position_features[i,28] = np.sin(y_norm*100*np.pi)+1
            position_features[i,29] = np.cos(y_norm*100*np.pi)+1
        return position_features.to(device)

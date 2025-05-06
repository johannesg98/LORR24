from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.data import Data, Batch
import numpy as np
import os


script_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNActor(nn.Module):
    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        in_channels += 6
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.act_dim = act_dim
        self.conv1 = NNConv(in_channels, self.out_channels, nn=nn.Sequential(nn.Linear(1,16), nn.ReLU(), nn.Linear(16,in_channels*self.out_channels)))
        self.lin1 = nn.Linear(in_channels+self.out_channels+1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.pos_feat = self.get_positions()
        self.edge_weights, self.edge_index_distancebased = self.get_edge_weights()

    def forward(self, state, edge_index, deterministic=False, return_dist=False, return_raw=False):
        positions = self.pos_feat.unsqueeze(0).expand(state.shape[0], -1, -1)
        state = torch.cat((state, positions), dim=-1)

        data_list = []
        for i in range(state.shape[0]):
            data_list.append(Data(x=state[i], edge_index=self.edge_index_distancebased, edge_attr=self.edge_weights))
        batch = Batch.from_data_list(data_list)

        out1 = F.relu(self.conv1(batch.x, batch.edge_index, edge_attr=batch.edge_attr))
        out1 = out1.reshape(-1, self.act_dim, self.out_channels)

        total_agents = state[...,1].sum(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, self.act_dim, -1)
        x = torch.cat((out1, total_agents, state), dim=-1)
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
        position_features = torch.zeros((nNodesss,6))
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
        return position_features.to(device)

    def get_edge_weights(self):
        NodeCostMatrix = torch.load(os.path.join(script_dir, "../data/NodeCostMatrix.pt"))
        nNodesss = NodeCostMatrix.shape[0]
        print("NodeCostMatrix Nodes: ", nNodesss)
        edge_limit = 0.5

        NodeCostMatrix = NodeCostMatrix/NodeCostMatrix.max()
        origin = []
        destination = []
        weight = []
        for o in range(nNodesss):
          for d in range(nNodesss):
            if NodeCostMatrix[o][d] < edge_limit:
              origin.append(o)
              destination.append(d)
              weight.append(NodeCostMatrix[o][d])

        edge_index_distancebased = torch.cat([torch.tensor([origin]), torch.tensor([destination])])
        weights = torch.tensor([weight])

        weights = (edge_limit-weights) / edge_limit


        return weights.squeeze(0).unsqueeze(-1).to(device), edge_index_distancebased.to(device)
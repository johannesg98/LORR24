from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.data import Data, Batch
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(os.path.abspath(__file__))

class GNNActor(nn.Module):
    def __init__(self, in_channels, hidden_size=32, act_dim=6, edge_limit=0.5, out_channel_fac=2):
        super().__init__()
        self.edge_limit = edge_limit
        in_channels += 6
        self.in_channels = in_channels
        self.out_channels = out_channel_fac * in_channels
        self.act_dim = act_dim
        # self.conv1 = NNConv(in_channels, self.out_channels, nn=nn.Sequential(nn.Linear(1,16), nn.ReLU(), nn.Linear(16,in_channels*self.out_channels)))
        self.conv1 = TransformerConv(in_channels, self.out_channels, edge_dim=1)
        self.lin1 = nn.Linear(in_channels+self.out_channels+1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.lin1_norm = nn.LayerNorm(hidden_size)
        self.lin2_norm = nn.LayerNorm(hidden_size)
        self.pos_feat = self.get_positions()
        self.edge_weights, self.edge_index_distancebased = self.get_edge_weights()
        self.max_storage = np.empty(0)



    def forward(self, state, edge_index, deterministic=False, return_dist=False, return_raw=False):
        deterministic=True
        if state.dim() == 3:
            positions = self.pos_feat.unsqueeze(0).expand(state.shape[0], -1, -1)
        elif state.dim() == 2:
            positions = self.pos_feat
        else:
            raise ValueError("State tensor must be 2D or 3D.")
        state = torch.cat((state, positions), dim=-1)

        if state.dim() == 3:
            state = state.reshape(-1, self.act_dim, self.in_channels)
            data_list = []
            for i in range(state.shape[0]):
                data_list.append(Data(x=state[i], edge_index=self.edge_index_distancebased, edge_attr=self.edge_weights))
            batch = Batch.from_data_list(data_list)

            out1 = F.relu(self.conv1(batch.x, batch.edge_index, edge_attr=batch.edge_attr))
        elif state.dim() == 2:
            out1 = F.relu(self.conv1(state, self.edge_index_distancebased, edge_attr=self.edge_weights))
            state = state.reshape(-1, self.act_dim, self.in_channels)
        
        out1 = out1.reshape(-1, self.act_dim, self.out_channels)
        total_agents = state[...,1].sum(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, self.act_dim, -1)
        x = torch.cat((out1, total_agents, state), dim=-1)
        
        x = F.leaky_relu(self.lin1_norm(self.lin1(x)))
        x = F.leaky_relu(self.lin2_norm(self.lin2(x)))
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


        ac_np = concentration.cpu().detach().numpy()#
        if ac_np.max(axis=-1) < 0.0000001:
            print("Max concentration < 0.0000001. Prob no conscious action taken.") 

        return action, log_prob
    
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
            position_features[i,2] = (np.sin(x_norm*2*np.pi)+1)/2
            position_features[i,3] = (np.cos(x_norm*2*np.pi)+1)/2
            position_features[i,4] = (np.sin(y_norm*2*np.pi)+1)/2
            position_features[i,5] = (np.cos(y_norm*2*np.pi)+1)/2
        return position_features.to(device)
    

    def get_edge_weights(self):
        NodeCostMatrix = torch.load(os.path.join(script_dir, "../../../nn_optimization/data/NodeCostMatrix.pt"))
        nNodesss = NodeCostMatrix.shape[0]
        print("NodeCostMatrix Nodes: ", nNodesss)

        NodeCostMatrix = NodeCostMatrix/NodeCostMatrix.max()
        origin = []
        destination = []
        weight = []
        for o in range(nNodesss):
          for d in range(nNodesss):
            if NodeCostMatrix[o][d] < self.edge_limit:
              origin.append(o)
              destination.append(d)
              weight.append(NodeCostMatrix[o][d])

        edge_index_distancebased = torch.cat([torch.tensor([origin]), torch.tensor([destination])])
        weights = torch.tensor([weight])

        weights = (self.edge_limit-weights) / self.edge_limit

        return weights.squeeze(0).unsqueeze(-1).to(device), edge_index_distancebased.to(device)

    


class GNNCritic(nn.Module):
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
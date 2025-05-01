from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.nn import GCNConv
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNActor(nn.Module):
    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels, add_self_loops=False)
        self.conv2 = GCNConv(in_channels, in_channels, add_self_loops=False)
        self.conv3 = GCNConv(in_channels, in_channels, add_self_loops=False)
        self.conv4 = GCNConv(in_channels, in_channels, add_self_loops=False)
        self.conv5 = GCNConv(in_channels, in_channels, add_self_loops=False)
        self.lin1 = nn.Linear(6*in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.new_edges = False

    def get_further_edges(self, edge_index):
        self.new_edges = True
        nNodes = max(max(edge_index[0]), max(edge_index[1])) + 1
        AdjacencyMatrix = np.zeros((nNodes, nNodes))
        for i in range(len(edge_index[0])):
            AdjacencyMatrix[edge_index[0][i]][edge_index[1][i]] = 1
            AdjacencyMatrix[edge_index[1][i]][edge_index[0][i]] = 1
        AM2 = np.zeros((nNodes, nNodes))
        for curr_node in range(nNodes):
          for node1 in range(nNodes):
            if AdjacencyMatrix[curr_node][node1] == 1:
              for node2 in range(nNodes):
                if AdjacencyMatrix[node1][node2] == 1:
                  if AdjacencyMatrix[curr_node][node2] == 0:
                    AM2[curr_node][node2] = 1
                    AM2[node2][curr_node] = 1
        AM3 = np.zeros((nNodes, nNodes))
        for curr_node in range(nNodes):
          for node2 in range(nNodes):
            if AM2[curr_node][node2] == 1:
              for node3 in range(nNodes):
                if AdjacencyMatrix[node2][node3] == 1:
                  if AM2[curr_node][node3] == 0 and AdjacencyMatrix[curr_node][node3] == 0:
                    AM3[curr_node][node3] = 1
                    AM3[node3][curr_node] = 1
        AM4 = np.zeros((nNodes, nNodes))
        for curr_node in range(nNodes):
          for node3 in range(nNodes):
            if AM3[curr_node][node3] == 1:
              for node4 in range(nNodes):
                if AdjacencyMatrix[node3][node4] == 1:
                  if AM3[curr_node][node4] == 0 and AM2[curr_node][node4] == 0 and AdjacencyMatrix[curr_node][node4] == 0:
                    AM4[curr_node][node4] = 1
                    AM4[node4][curr_node] = 1
        AM5 = np.zeros((nNodes, nNodes))
        for curr_node in range(nNodes):
          for node4 in range(nNodes):
            if AM4[curr_node][node4] == 1:
              for node5 in range(nNodes):
                if AdjacencyMatrix[node4][node5] == 1:
                  if AM4[curr_node][node5] == 0 and AM3[curr_node][node5] == 0 and AM2[curr_node][node5] == 0 and AdjacencyMatrix[curr_node][node5] == 0:
                    AM5[curr_node][node5] = 1
                    AM5[node5][curr_node] = 1
        self.edge_index2 = self.get_edge_index(AM2).to(device)
        self.edge_index3 = self.get_edge_index(AM3).to(device)
        self.edge_index4 = self.get_edge_index(AM4).to(device)
        self.edge_index5 = self.get_edge_index(AM5).to(device)
        
    
    def get_edge_index(self, AdjacencyMatrix):
        origin = []
        destination = []
        for o in range(AdjacencyMatrix.shape[0]):
            for d in range(AdjacencyMatrix.shape[0]):
                if AdjacencyMatrix[o][d] == 1:
                    origin.append(o)
                    destination.append(d)

        edge_index = torch.cat([torch.tensor([origin]), torch.tensor([destination])])
        return edge_index
        


    def forward(self, state, edge_index, deterministic=False, return_dist=False, return_raw=False):
        if not self.new_edges:
            self.get_further_edges(edge_index)
        out1 = F.relu(self.conv1(state, edge_index))
        out2 = F.relu(self.conv2(state, self.edge_index2))
        out3 = F.relu(self.conv3(state, self.edge_index3))
        out4 = F.relu(self.conv3(state, self.edge_index4))
        out5 = F.relu(self.conv3(state, self.edge_index5))

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


class GNNActorOld(nn.Module):
    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, deterministic=False, return_dist=False, return_raw=False):
        out = F.relu(self.conv1(state, edge_index))
        if torch.isnan(out).any():
            print("NaN values detected in out!")
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
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
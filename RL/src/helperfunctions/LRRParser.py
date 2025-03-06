import math
import torch
from torch_geometric.data import Data

class LRRParser:
    def __init__(self, env, cfg):
        self.input_size = cfg.input_size
        self.nNodes = env.nNodes
        self.agent_scale_fac = 1/math.ceil(env.nAgents/env.nNodes*4)
        self.task_scale_fac = 1/math.ceil(env.nTasks/env.nNodes*4)
        self.edge_index = self.get_edge_index(env)
        
        
    def parse_obs(self, obs):
        x = torch.cat((
            torch.clip(torch.tensor(obs["agents_per_node"])*self.agent_scale_fac, 0, 1).view(1, self.nNodes).float(),
            torch.clip(torch.tensor(obs["free_agents_per_node"])*self.agent_scale_fac, 0, 1).view(1, self.nNodes).float(),
            torch.clip(torch.tensor(obs["free_tasks_per_node"])*self.task_scale_fac, 0, 1).view(1, self.nNodes).float()
        )
        ,dim=0).view(self.input_size, self.nNodes).T

        data = Data(x, self.edge_index)
        return data


    def get_edge_index(self, env):
        origin = []
        destination = []
        for o in range(len(env.AdjacencyMatrix)):
            for d in range(len(env.AdjacencyMatrix)):
                if env.AdjacencyMatrix[o][d] == 1:
                    origin.append(o)
                    destination.append(d)

        edge_index = torch.cat([torch.tensor([origin]), torch.tensor([destination])])
        return edge_index


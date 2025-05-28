import math
import torch
from torch_geometric.data import Data
import numpy as np

class LRRParser:
    def __init__(self, env, cfg):
        self.cfg = cfg
        self.input_size = cfg.input_size
        self.nNodes = env.nNodes
        self.normalise_obs = cfg.normalise_obs
        self.agent_scale_fac = 1/math.ceil(env.nAgents/env.nNodes*4)
        self.task_scale_fac = 1/math.ceil(env.nTasks/env.nNodes*4)
        self.edge_index = self.get_edge_index(env)
        self.MP_edge_weights = env.MP_edge_weights
        self.MP_nEdges = len(self.MP_edge_weights)
        self.node_positions = env.node_positions
        
        
    def parse_obs(self, obs):
        if self.normalise_obs:
            x = torch.cat((
                # per node stuff
                torch.clip(torch.tensor(obs["agents_per_node"])*self.agent_scale_fac, 0, 1).view(1, self.nNodes).float(),                                   # agents per node
                torch.clip(torch.tensor(obs["free_agents_per_node"])*self.agent_scale_fac, 0, 1).view(1, self.nNodes).float(),                              # free agents per node
                torch.clip(torch.tensor(obs["free_tasks_per_node"])*self.task_scale_fac, 0, 1).view(1, self.nNodes).float(),                                # free tasks per node
                torch.tensor(obs["congestion_ratio_per_node"]).view(1, self.nNodes).float(),                                                                # congestion ratio per node
                torch.tensor(self.node_positions).view(6, self.nNodes).float(),                                                                             # 6 x node positions
                (torch.tensor(obs["distance_until_agent_available_per_node"])/self.cfg.distance_until_agent_avail_MAX).view(1, self.nNodes).float(),        # distance until agent available per node
                # torch.clip(torch.tensor(obs["agents_available_next_steps_per_node"])*self.agent_scale_fac, 0, 1).T.view(self.cfg.distance_until_agent_avail_MAX, self.nNodes).float(),                    # number agents available per node at next steps
                torch.clip(torch.tensor(obs["contains_closest_task_per_node"])*self.agent_scale_fac, 0, 1).view(1, self.nNodes).float(),                    # indicates how many free agents have there closest task in this node

                # same for every node
                (torch.tensor(obs["free_agents_per_node"])*self.agent_scale_fac).sum().view(1, 1).expand(1, self.nNodes).float(),                           # total free agents world
                (torch.tensor(obs["time"])/self.cfg.max_steps).view((1,1)).expand(1, self.nNodes).float(),                                                  # time         
                torch.tensor((np.sin(obs["time"]/100*2*np.pi)+1)/2).view((1,1)).expand(1, self.nNodes).float(),                                           # sin(time)     
                torch.tensor((np.cos(obs["time"]/100*2*np.pi)+1)/2).view((1,1)).expand(1, self.nNodes).float(),                                           # cos(time)    

            )
            
            ,dim=0).view(self.input_size, self.nNodes).T
        else:
            x = torch.cat((
                torch.tensor(obs["agents_per_node"]).view(1, self.nNodes).float(),
                torch.tensor(obs["free_agents_per_node"]).view(1, self.nNodes).float(),
                torch.tensor(obs["free_tasks_per_node"]).view(1, self.nNodes).float(),
                torch.tensor(obs["congestion_ratio_per_node"]).view(1, self.nNodes).float(),
                torch.tensor(self.node_positions).view(6, self.nNodes).float(),
                torch.tensor(obs["distance_until_agent_available_per_node"]).view(1, self.nNodes).float(),
                # torch.tensor(obs["agents_available_next_steps_per_node"]).T.view(self.cfg.distance_until_agent_avail_MAX, self.nNodes).float(),
                torch.tensor(obs["contains_closest_task_per_node"]).view(1, self.nNodes).float(),

                torch.tensor(obs["free_agents_per_node"]).sum().view(1, 1).expand(1, self.nNodes).float(),
                torch.tensor(obs["time"]).reshape((1,1)).expand(1, self.nNodes).float(),
                torch.tensor((np.sin(obs["time"]/100*2*np.pi)+1)/2).view((1,1)).expand(1, self.nNodes).float(),                                           # sin(time)     
                torch.tensor((np.cos(obs["time"]/100*2*np.pi)+1)/2).view((1,1)).expand(1, self.nNodes).float(), 
            )
            
            ,dim=0).view(self.input_size, self.nNodes).T
        


        # edge attributes
        if self.cfg.use_message_passing:
            edge_atributes = torch.cat((
                torch.tensor(self.MP_edge_weights).view(1, self.MP_nEdges).float(),
                torch.tensor(obs["congestion_ratio_per_edge"]).T.view(3, self.MP_nEdges).float(),
                # torch.tensor(obs["congestion_ratio_next_steps_per_edge"]).T.view(self.cfg.distance_until_agent_avail_MAX, self.MP_nEdges).float(),
                torch.tensor(obs["closest_task_connection_per_MP_edge"]).view(1, self.MP_nEdges).float(),
                torch.tensor(obs["agents_waiting_per_edge"]).view(1, self.MP_nEdges).float()

            ), dim=0).view(self.cfg.edge_feature_dim, self.MP_nEdges).T
        else:
            edge_atributes = torch.tensor([])


        data = Data(x, self.edge_index, edge_attr=edge_atributes)
        return data


    def get_edge_index(self, env):
        if self.cfg.use_message_passing:
            edge_index = torch.cat([torch.tensor([env.MP_edge_index[0]]), torch.tensor([env.MP_edge_index[1]])])
        else:
            origin = []
            destination = []
            ad = env.AdjacencyMatrix
            for o in range(len(ad)):
                for d in range(len(ad)):
                    if ad[o][d] == 1:
                        origin.append(o)
                        destination.append(d)

            edge_index = torch.cat([torch.tensor([origin]), torch.tensor([destination])])
        return edge_index
    
  
        

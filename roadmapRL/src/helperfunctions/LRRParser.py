import math
import torch
from torch_geometric.data import Data
import numpy as np

class LRRParser:
    def __init__(self, env, cfg):
        self.cfg = cfg
        self.input_size = cfg.input_size
        self.nNodes = env.nRoadmapNodes
        self.normalise_obs = cfg.normalise_obs
        self.agent_scale_fac = 1/3
        self.edge_index, self.edge_type, self.nEdges = self.get_edge_index(env)
        self.node_positions = env.roadmapNodePositions
        
        
    def parse_obs(self, obs):
        if self.normalise_obs:
            x = torch.cat((
                # per node stuff
                torch.clip(torch.tensor(obs["RM_agents_per_node"])*self.agent_scale_fac, 0, 1).view(1, self.nNodes).float(),                                  
                torch.clip(torch.tensor(obs["RM_free_agents_per_node"])*self.agent_scale_fac, 0, 1).view(1, self.nNodes).float(),      
                torch.clip(torch.tensor(obs["RM_agents_per_node_in_dir"])*self.agent_scale_fac, 0, 1).view(1, self.nNodes).float(),      
                torch.clip(torch.tensor(obs["RM_agents_per_node_90_dir"])*self.agent_scale_fac, 0, 1).view(1, self.nNodes).float(),      
                torch.clip(torch.tensor(obs["RM_agents_per_node_op_dir"])*self.agent_scale_fac, 0, 1).view(1, self.nNodes).float(),      
                torch.tensor(self.node_positions).view(6, self.nNodes).float(),       

                # same for every node
                (torch.tensor(obs["time"])/self.cfg.max_steps).view((1,1)).expand(1, self.nNodes).float(),                                                        
                torch.tensor((np.sin(obs["time"]/100*2*np.pi)+1)/2).view((1,1)).expand(1, self.nNodes).float(),                                             
                torch.tensor((np.cos(obs["time"]/100*2*np.pi)+1)/2).view((1,1)).expand(1, self.nNodes).float()                                     

            )
            
            ,dim=0).view(self.input_size, self.nNodes).T
        else:
            x = torch.cat((
                # per node stuff
                torch.tensor(obs["RM_agents_per_node"]).view(1, self.nNodes).float(),                                  
                torch.tensor(obs["RM_free_agents_per_node"]).view(1, self.nNodes).float(),      
                torch.tensor(obs["RM_agents_per_node_in_dir"]).view(1, self.nNodes).float(),      
                torch.tensor(obs["RM_agents_per_node_90_dir"]).view(1, self.nNodes).float(),      
                torch.tensor(obs["RM_agents_per_node_op_dir"]).view(1, self.nNodes).float(),      
                torch.tensor(self.node_positions).view(6, self.nNodes).float(),       

                # same for every node
                torch.tensor(obs["time"]).view((1,1)).expand(1, self.nNodes).float(),                                                        
                torch.tensor((np.sin(obs["time"]/100*2*np.pi)+1)/2).view((1,1)).expand(1, self.nNodes).float(),                                             
                torch.tensor((np.cos(obs["time"]/100*2*np.pi)+1)/2).view((1,1)).expand(1, self.nNodes).float() 
            )
            
            ,dim=0).view(self.input_size, self.nNodes).T
        


        # edge attributes
        edge_atributes = torch.cat((
            self.edge_type.view(4, self.nEdges).float(),
            torch.tensor(obs["RM_agents_per_edge_in_dir"]).view(1, self.nEdges).float(),
            torch.tensor(obs["RM_agents_per_edge_90_dir"]).view(1, self.nEdges).float(),
            torch.tensor(obs["RM_agents_per_edge_op_dir"]).view(1, self.nEdges).float()

        ), dim=0).view(self.cfg.edge_feature_dim, self.nEdges).T
        

        data = Data(x, self.edge_index, edge_attr=edge_atributes)
        return data


    def get_edge_index(self, env):
        origin = []
        destination = []
        edge_type = [[],[],[],[]]
        ad = env.roadmapAdjacencyMatrix
        for o in range(self.nNodes):
            for d in range(self.nNodes):
                if ad[o][d] > 0:
                    origin.append(o)
                    destination.append(d)
                    for i in range(4):
                        if ad[o][d] == i+1:
                            edge_type[i].append(1)
                        else:
                            edge_type[i].append(0)

        edge_index = torch.cat([torch.tensor([origin]), torch.tensor([destination])])
        edge_type = torch.tensor(edge_type)
        nEdges = edge_index.shape[1]
        return edge_index, edge_type, nEdges
    
  
        

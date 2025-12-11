#pragma once
#include "common.h"
#include "SharedEnv.h"

class RoadmapGraph{
public:
    RoadmapGraph(){};
    RoadmapGraph(SharedEnvironment* env, std::vector<std::vector<int>>& fixed_roadmap);
    

    int nNodes, nEdges;
    std::vector<std::vector<int>> node_to_locs;
    std::vector<int> loc_to_node;
    std::vector<int> node_direction;                            // node_idx -> direction of the node (path) 0: east, 1: south, 2: west, 3: north
    std::vector<std::vector<int>> node_to_cost_locs;            // node_idx -> vector of locations where decided cost will be applied
    
    std::vector<int> edge_to_loc;    
    std::vector<std::pair<int, int>> edge_to_node_start_end;
    std::vector<std::vector<int>> loc_to_edges;

    std::vector<std::vector<int>> AdjacencyMatrix;              // 0: no edge, 1: edge from end of i to start of j, 2: edge from start of i to end of j, 3: edge from end of i to end of j, 4: edge from start of i to start of j

private:
    std::tuple<int, int, int, int, int> get_directed_node(SharedEnvironment* env, std::vector<std::vector<int>>& fixed_roadmap, int i, int j);
};




#include "RoadmapGraph.h"

// Inverts the inner node graph. Nodes become edges and edges become nodes.
RoadmapGraph::RoadmapGraph(SharedEnvironment* env, std::vector<std::vector<int>>& fixed_roadmap){
    loc_to_node.resize(env->map.size(), -1);
    loc_to_edges.resize(env->map.size(), std::vector<int>());
    
    std::vector<int> node_to_global_node_start;
    std::vector<int> node_to_global_node_end;
    int start_loc, end_loc, dir, start_global_node, end_global_node;

    // Iterate through all inner global edges and create directed roadmap nodes for each inner global edge
    int node_id = 0;
    for (int i = 0; i < env->nodes->nInnerNodes; i++) {
        for (int j = 0; j < i; j++) {
            if (env->nodes->AdjacencyMatrix[i][j] == 1) {
                std::tie(start_loc, end_loc, dir, start_global_node, end_global_node) = get_directed_node(env, fixed_roadmap, i, j);

                node_to_global_node_end.push_back(end_global_node);
                node_to_global_node_start.push_back(start_global_node);

                node_direction.push_back(dir);
                
                node_to_locs.push_back(std::vector<int>());
                node_to_cost_locs.push_back(std::vector<int>());
                int curr_loc = start_loc;
                while (curr_loc != end_loc) {
                    if (dir == 0) curr_loc++;
                    else if (dir == 1) curr_loc += env->cols;
                    else if (dir == 2) curr_loc--;
                    else if (dir == 3) curr_loc -= env->cols;
                    if (curr_loc != end_loc) {
                        node_to_locs[node_id].push_back(curr_loc);
                        loc_to_node[curr_loc] = node_id;
                    }
                    node_to_cost_locs[node_id].push_back(curr_loc);
                }
                node_id++;
            }
        }
    }
    nNodes = node_id;

    // Create roadmap edges (at the locations of the global inner nodes)
    AdjacencyMatrix.resize(nNodes, std::vector<int>(nNodes, 0));
    int edge_loc;
    int edge_id = 0;
    for (int i = 0; i < nNodes; i++) {
        for (int j = 0; j < nNodes; j++) {
            if (i == j) continue;
            if (node_to_global_node_end[i] == node_to_global_node_start[j]) {
                // Edge from end of i to start of j
                AdjacencyMatrix[i][j] = 1;
                edge_loc = env->nodes->locations[node_to_global_node_start[j]];
                edge_to_loc.push_back(edge_loc);
                loc_to_edges[edge_loc].push_back(edge_id);
                edge_to_node_start_end.push_back({i, j});
                edge_id++;
            }
            else if (node_to_global_node_start[i] == node_to_global_node_end[j]) {
                // Edge from start of i to end of j
                AdjacencyMatrix[i][j] = 2;
                edge_loc = env->nodes->locations[node_to_global_node_end[j]];
                edge_to_loc.push_back(edge_loc);
                loc_to_edges[edge_loc].push_back(edge_id);
                edge_to_node_start_end.push_back({i, j});
                edge_id++;
            }
            else if (node_to_global_node_end[i] == node_to_global_node_end[j]) {
                // Edge from end of i to end of j
                AdjacencyMatrix[i][j] = 3;
                edge_loc = env->nodes->locations[node_to_global_node_end[j]];
                edge_to_loc.push_back(edge_loc);
                loc_to_edges[edge_loc].push_back(edge_id);
                edge_to_node_start_end.push_back({i, j});
                edge_id++;
            }
            else if (node_to_global_node_start[i] == node_to_global_node_start[j]) {
                // Edge from start of i to start of j
                AdjacencyMatrix[i][j] = 4;
                edge_loc = env->nodes->locations[node_to_global_node_start[j]];
                edge_to_loc.push_back(edge_loc);
                loc_to_edges[edge_loc].push_back(edge_id);
                edge_to_node_start_end.push_back({i, j});
                edge_id++;
            }
        }
    }
    nEdges = edge_id;
}


std::tuple<int, int, int, int, int> RoadmapGraph::get_directed_node(SharedEnvironment* env, std::vector<std::vector<int>>& fixed_roadmap, int i, int j) {
    int i_loc = env->nodes->locations[i];
    int j_loc = env->nodes->locations[j];
    int middle_loc, start_loc, end_loc, dir, start_global_node, end_global_node;
    if (j_loc >= i_loc + env->cols){
        middle_loc = i_loc + env->cols;
        if (fixed_roadmap[middle_loc][3] > 0) {
            start_loc = i_loc;
            end_loc = j_loc;
            dir = 1; // south
        }
        else if( fixed_roadmap[middle_loc][1] > 0) {
            start_loc = j_loc;
            end_loc = i_loc;
            dir = 3; // north
        }
        else {
            std::cerr << "Error: No roadmap direction for edge between nodes " << i << " and " << j << std::endl;
        }
    }
    else if (j_loc <= i_loc - env->cols){
        middle_loc = i_loc - env->cols;
        if (fixed_roadmap[middle_loc][1] > 0) {
            start_loc = i_loc;
            end_loc = j_loc;
            dir = 3; // north
        }
        else if( fixed_roadmap[middle_loc][3] > 0) {
            start_loc = j_loc;
            end_loc = i_loc;
            dir = 1; // south
        }
        else {
            std::cerr << "Error: No roadmap direction for edge between nodes " << i << " and " << j << std::endl;
        }
    }
    else if (j_loc > i_loc){
        middle_loc = i_loc + 1;
        if (fixed_roadmap[middle_loc][2] > 0) {
            start_loc = i_loc;
            end_loc = j_loc;
            dir = 0; // east
        }
        else if( fixed_roadmap[middle_loc][0] > 0) {
            start_loc = j_loc;
            end_loc = i_loc;
            dir = 2; // west
        }
        else {
            std::cerr << "Error: No roadmap direction for edge between nodes " << i << " and " << j << std::endl;
        }
    }
    else if (j_loc < i_loc){
        middle_loc = i_loc - 1;
        if (fixed_roadmap[middle_loc][0] > 0) {
            start_loc = i_loc;
            end_loc = j_loc;
            dir = 2; // west
        }
        else if( fixed_roadmap[middle_loc][2] > 0) {
            start_loc = j_loc;
            end_loc = i_loc;
            dir = 0; // east
        }
        else {
            std::cerr << "Error: No roadmap direction for edge between nodes " << i << " and " << j << std::endl;
        }
    }
    else {
        std::cerr << "Error: Invalid node locations for edge between nodes " << i << " and " << j << std::endl;
    }
    if (start_loc == i_loc) {
        start_global_node = i;
        end_global_node = j;
    } else {
        start_global_node = j;
        end_global_node = i;
    }
    return std::make_tuple(start_loc, end_loc, dir, start_global_node, end_global_node);
}


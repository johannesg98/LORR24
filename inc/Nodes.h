#pragma once
#include "common.h"

class Nodes{
public:
    Nodes(string fname);
    void printData();

    int rows, cols, nNodes, nInnerNodes;
    std::vector<int> locations;     // node_idx -> location_id of node center
    std::vector<int> regions;       // location_id (of any location in the node region)-> node_idx
    std::vector<std::vector<int>> AdjacencyMatrix;
};
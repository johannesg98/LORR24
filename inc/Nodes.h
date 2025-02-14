#include "common.h"

class Nodes{
public:
    Nodes(string fname);
    void printData();

    int rows, cols, nNodes;
    std::vector<int> nodes;
    std::vector<int> regions;
    std::vector<std::vector<int>> AdjacencyMatrix;
};
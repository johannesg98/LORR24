#include "Nodes.h"


Nodes::Nodes(string fname){

    std::ifstream myfile ((fname).c_str());
    if (!myfile.is_open())
    {
        cout << "Node file " << fname << " does not exist. " << std::endl;
        exit(-1);
    }

     // Read height, width, and nNodes
     std::string line;
     myfile >> line >> rows;
     myfile >> line >> cols;
     myfile >> line >> nNodes;
     myfile.ignore(); // Move to the next line
 
     // Read nodes
     std::getline(myfile, line);
     std::stringstream ss(line);
     int value;
     while (ss >> value) {
         nodes.push_back(value);
         if (ss.peek() == ',') ss.ignore(); // Ignore commas
     }
 
     // Read regions (rows x cols matrix)
     regions.resize(rows*cols);
     for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; ++j) {
             myfile >> regions[i*cols+j];
             if (myfile.peek() == ',') myfile.ignore();
         }
     }
 
     // Read Adjacency Matrix (nNodes x nNodes matrix)
     AdjacencyMatrix.resize(nNodes, std::vector<int>(nNodes));
     for (int i = 0; i < nNodes; ++i) {
         for (int j = 0; j < nNodes; ++j) {
             myfile >> AdjacencyMatrix[i][j];
             if (myfile.peek() == ',') myfile.ignore();
         }
     }
 }


 void Nodes::printData(){
    std::cout << "Rows: " << rows  << " Cols: " << cols << " nNodes: " << nNodes << std::endl; 
    std::cout << "Regions: " << std::endl; 
    for (int num : regions) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    std::cout << "Nodes: " << std::endl; 
    for (int num : nodes) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    std::cout << "Adjacency Matrix:\n";
    for (const auto& row : AdjacencyMatrix) {
        for (int value : row) {
            std::cout << std::setw(3) << value << " ";  // Format with spacing
        }
        std::cout << std::endl;
    }
 }


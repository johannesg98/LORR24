#include "edge_features.h"

namespace edgeFeatures{

int manhatten(int a, int b, SharedEnvironment* env){
    int ax = a % env->cols;
    int ay = a / env->cols;
    int bx = b % env->cols;
    int by = b / env->cols;
    return abs(ax - bx) + abs(ay - by);
}

Direction get_direction(int node, int cameFrom, SharedEnvironment* env){
    if (node - cameFrom == -env->cols) return north;
    if (node - cameFrom == env->cols) return south;
    if (node - cameFrom == -1) return west;
    if (node - cameFrom == 1) return east;
    std::cerr << "Error: Invalid direction from " << cameFrom << " to " << node << std::endl;
}


std::vector<PathNode> astar(int start, int goal, SharedEnvironment* env, DefaultPlanner::Neighbors* ns){
    std::unordered_map<int, int> cameFrom;
    std::unordered_map<int, int> distance;
    std::priority_queue<QueueNode, std::vector<QueueNode>, std::greater<QueueNode>> openSet;

    distance[start] = 0;
    openSet.push({start, 0});

    while (!openSet.empty()) {
        QueueNode current = openSet.top();
        openSet.pop();

        if (current.loc == goal) {
            std::vector<PathNode> path;
            int node = goal;
            while (node != start) {
                Direction dir = get_direction(node, cameFrom[node], env);
                path.push_back({node, dir});                               
                node = cameFrom[node];
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        for (int neighbor : (*ns)[current.loc]) {
            int newCost = distance[current.loc] + 1;
            if (distance.find(neighbor) == distance.end() || newCost < distance[neighbor]) {
                distance[neighbor] = newCost;
                cameFrom[neighbor] = current.loc;
                openSet.push({neighbor, newCost + manhatten(neighbor, goal, env)});
            }
        }        
    }
    std::cerr << "A* failed to find a path from " << start << " to " << goal << std::endl;

}




}
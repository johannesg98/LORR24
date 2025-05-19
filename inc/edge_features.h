#pragma once
#include "common.h"
#include "SharedEnv.h"
#include "Types.h"


namespace edgeFeatures
{
    enum Direction {
        east = 0,
        south = 1,
        west = 2,
        north = 3
    };

    struct PathNode {
        int loc;
        Direction dir;
    };

    struct QueueNode {
        int loc;
        int f;  // f = dist + h
        bool operator>(const QueueNode& other) const {
            return f > other.f;
        }
    };

    int manhatten(int a, int b, SharedEnvironment* env);
    Direction get_direction(int node, int cameFrom, SharedEnvironment* env);
    std::vector<PathNode> astar(int start, int goal, SharedEnvironment* env, DefaultPlanner::Neighbors* ns);
}
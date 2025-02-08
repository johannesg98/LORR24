#ifndef EXTENDED_BASE_SYSTEM_H
#define EXTENDED_BASE_SYSTEM_H

#include "../inc/CompetitionSystem.h"

class ExtendedBaseSystem : public BaseSystem {
public:
    // Constructor that calls the BaseSystem constructor
    ExtendedBaseSystem(Grid &grid, Entry* planner, std::vector<int>& start_locs, 
                       std::vector<list<int>>& tasks, ActionModelWithRotate* model)
    : BaseSystem(grid, planner, start_locs, tasks, model) {}

    // New function added in this derived class
    void initializeExtendedBaseSystem(int simulation_time);
    bool step();
};

#endif // EXTENDED_BASE_SYSTEM_H

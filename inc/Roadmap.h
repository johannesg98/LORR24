#pragma once
#include "common.h"
#include "SharedEnv.h"
#include <RoadmapGraph.h>

class Roadmap{
public:
    Roadmap(){};
    Roadmap(SharedEnvironment* env,const std::string& fixed_roadmap_path);
    
    bool updated_last_step = false;
    RoadmapGraph graph;

    std::vector<std::vector<int>> fixed_roadmap;
    std::vector<std::vector<int>> activated_roadmap;

    void update_roadmap(SharedEnvironment* env, std::vector<int>& node_activation);
    
private:
    
};


std::vector<std::vector<int>> load_fixed_roadmap(SharedEnvironment* env,const std::string fname);

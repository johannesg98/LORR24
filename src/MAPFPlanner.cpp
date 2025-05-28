#include <random>
#include <Entry.h>

//default planner includes
#include "planner.h"
#include "const.h"

/**
 * Initialises the MAPF planner with a given time limit for preprocessing.
 * 
 * This function call the planner initialize function with a time limit fore preprocessing.
 * 
 * @param preprocess_time_limit The total time limit allocated for preprocessing (in milliseconds).
 */
void MAPFPlanner::initialize(int preprocess_time_limit)
{
    // use the remaining entry time limit (after task scheduling) for path planning, -PLANNER_TIMELIMIT_TOLERANCE for timing error tolerance;
    int limit = preprocess_time_limit - std::chrono::duration_cast<milliseconds>(std::chrono::steady_clock::now() - env->plan_start_time).count() - DefaultPlanner::PLANNER_TIMELIMIT_TOLERANCE;
    
    if (planner_type.empty()){
        // fallback for LRR standard build
        DefaultPlanner::initialize(limit, env);
    }
    else if (planner_type == "default"){
        DefaultPlanner::initialize(limit, env);
    }
    else{
        std::cerr << "Unknown planner type: " << planner_type << std::endl;
        exit(1);
    }
    
    return;
}

/**
 * Plans a path using default planner
 * 
 * This function performs path planning within the timelimit given, and call the plan function in default planner.
 * The planned actions are output to the provided actions vector.
 * 
 * @param time_limit The time limit allocated for planning (in milliseconds).
 * @param actions A reference to a vector that will be populated with the planned actions (next action for each agent).
 */
void MAPFPlanner::plan(int time_limit,vector<Action> & actions) 
{
    
    int limit;
    if (guarantee_planner_time){
        // fix time for RL, independent on how much time already passed to guarantee full execution with comparable LNS duration
        limit = time_limit/2;
    }
    else{
        // Default in LRR: use the remaining time after task schedule for path planning, -PLANNER_TIMELIMIT_TOLERANCE for timing error tolerance;
        limit = time_limit - std::chrono::duration_cast<milliseconds>(std::chrono::steady_clock::now() - env->plan_start_time).count() - DefaultPlanner::PLANNER_TIMELIMIT_TOLERANCE;
    }

    std::cout << "Time (left) for planning: " << limit << " ms " << std::endl;
    auto now = std::chrono::high_resolution_clock::now();

    if (planner_type.empty()){
        // fallback for LRR standard build
        DefaultPlanner::plan(limit, actions, env);
    }
    else if (planner_type == "default"){
        DefaultPlanner::plan(limit, actions, env);
    }
    else{
        std::cerr << "Unknown planner type: " << planner_type << std::endl;
        exit(1);
    }
    

    std::chrono::duration<double> passed = std::chrono::high_resolution_clock::now() - now;
    std::cout << "Time planning used: " << passed.count() << " seconds" << std::endl;

    

    return;
}

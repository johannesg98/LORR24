#include "ExtendedBaseSystem.h"
#include <iostream>

void ExtendedBaseSystem::initializeExtendedBaseSystem(int simulation_time) {
    initialize();
    this->simulation_time = simulation_time;
}

bool ExtendedBaseSystem::step(){

    bool done = false;
    sync_shared_env();

    auto start = std::chrono::steady_clock::now();

    env->plan_start_time = std::chrono::steady_clock::now();

    planner->compute(plan_time_limit, proposed_actions, proposed_schedule);

    

    auto end = std::chrono::steady_clock::now();
    auto diff = end-start;
    planner_times.push_back(std::chrono::duration<double>(diff).count());

    if (simulator.get_curr_timestep() >= simulation_time){
        done = true;
        return done;
    }

    for (int a = 0; a < num_of_agents; a++){
        if (!env->goal_locations[a].empty())
            solution_costs[a]++;
    }

    vector<State> curr_states = simulator.move(proposed_actions);

    task_manager.update_tasks(curr_states, proposed_schedule, simulator.get_curr_timestep());

    logger->log_info("Step done.", simulator.get_curr_timestep());

    return done;
}

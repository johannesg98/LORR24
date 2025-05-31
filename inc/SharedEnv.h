#pragma once
#include "States.h"
#include "Grid.h"
#include "nlohmann/json.hpp"
#include "Tasks.h"
#include <unordered_map>
#include <unordered_set>

//RL added stuff
#include <Nodes.h>
#include <BacktrackBundle.h>


typedef std::chrono::steady_clock::time_point TimePoint;
typedef std::chrono::milliseconds milliseconds;
typedef std::unordered_map<int, Task> TaskPool;

class Roadmap; // for forward declaration

class SharedEnvironment
{
public:
    int num_of_agents;
    int rows;
    int cols;
    std::string map_name;
    std::vector<int> map;
    std::string file_storage_path;

    // goal locations for each agent
    // each task is a pair of <goal_loc, reveal_time>
    vector< vector<pair<int, int> > > goal_locations;

    int curr_timestep = 0;
    vector<State> curr_states;

    TaskPool task_pool; // task_id -> Task
    vector<int> new_tasks; // task ids of tasks that are newly revealed in the current timestep
    vector<int> new_freeagents; // agent ids of agents that are newly free in the current timestep
    vector<int> curr_task_schedule; // the current scheduler, agent_id -> task_id

    // plan_start_time records the time point that plan/initialise() function is called; 
    // It is a convenient variable to help planners/schedulers to keep track of time.
    // plan_start_time is updated when the simulation system call the entry plan function, its type is std::chrono::steady_clock::time_point
    TimePoint plan_start_time;

    //RL added stuff
    std::unique_ptr<Nodes> nodes;
    double Astar_reward = 0;
    double idle_agents_reward = 0;
    double tasks_assigned_reward = 0;
    double dist_reward = 0;
    std::vector<int> task_search_durations;
    std::vector<int> task_distances;
    std::vector<BacktrackBundle> backtrack_bundles_first_errand;
    std::unordered_map<int,std::vector<int>> backtrack_times_first_errand; //starttime, traveltimes
    std::unordered_map<int,double> backtrack_rewards_first_errand; //starttime, reward
    std::vector<BacktrackBundle> backtrack_bundles_whole_task;
    std::unordered_map<int,std::vector<int>> backtrack_times_whole_task;
    std::unordered_map<int,double> backtrack_rewards_whole_task;
    std::unordered_map<int,std::vector<float>> action_rl;
    bool use_dummy_goals_for_idle_agents = true;

    // Roadmap stuff
    Roadmap* roadmap = nullptr;
    int roadmap_reward_first_distance = 0;
    


    SharedEnvironment(){}
};

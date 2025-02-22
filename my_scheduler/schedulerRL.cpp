#include "schedulerRL.h"

namespace schedulerRL{

std::mt19937 mt;
std::unordered_set<int> free_agents;
std::unordered_set<int> free_tasks;


void schedule_initialize(int preprocess_time_limit, SharedEnvironment* env)
{   
    free_agents.clear();
    free_tasks.clear();

    // cout<<"schedule initialise limit" << preprocess_time_limit<<endl;
    DefaultPlanner::init_heuristics(env);
    mt.seed(0);
    return;
}

void schedule_plan(int time_limit, std::vector<int> & proposed_schedule,  SharedEnvironment* env, const std::unordered_map<std::string, pybind11::object>& action_dict)
{
    //use at most half of time_limit to compute schedule, -10 for timing error tolerance
    //so that the remainning time are left for path planner
    TimePoint endtime = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_limit);
    // cout<<"schedule plan limit" << time_limit <<endl;

    // the default scheduler keep track of all the free agents and unassigned (=free) tasks across timesteps
    free_agents.insert(env->new_freeagents.begin(), env->new_freeagents.end());
    free_tasks.insert(env->new_tasks.begin(), env->new_tasks.end());

    int min_task_i, min_task_makespan, dist, agent_loc, task_loc, count, node_id;
    clock_t start = clock();
    count = 0;

    
    std::map<pair<int,int>, int> reb_action = action_dict.at("reb_action").cast<std::map<pair<int,int>, int>>();

    // create a list of tasks per node
    std::vector<std::vector<int>> tasks_per_node(env->nodes->nNodes);
    for (int task_id = 0; task_id < env->task_pool.size(); task_id++){
        task_loc = env->task_pool[task_id].locations[0];
        node_id = env->nodes->regions[task_loc];
        tasks_per_node[node_id].push_back(task_id);
    }
    std::vector<std::vector<int>> free_agents_per_node(env->nodes->nNodes);
    for (int agent_id : free_agents){
        agent_loc = env->curr_states[agent_id].location;
        node_id = env->nodes->regions[agent_loc];
        free_agents_per_node[node_id].push_back(agent_id);
    }
    bool timeout = false;
    for (int goal_node = 0; goal_node < env->nodes->nNodes && !timeout; goal_node++){
        std::vector<int> start_nodes;
        for (int start_node = 0; start_node < env->nodes->nNodes; start_node++){
            if (start_node != goal_node){
                if(reb_action[{start_node, goal_node}] == 1){
                    start_nodes.push_back(start_node);
                }
            }
        }
        if (start_nodes.size() > 0 && tasks_per_node[goal_node].size() > 0){
            for (int agent_id : start_nodes){
                min_task_i = -1;
                min_task_makespan = INT_MAX;
                agent_loc = env->curr_states[agent_id].location;
                for (int task_id : tasks_per_node[goal_node]){
                    task_loc = env->task_pool[task_id].locations[0];
                    dist = DefaultPlanner::get_h(env, agent_loc, task_loc);
                    if (dist < min_task_makespan){
                        min_task_i = task_id;
                        min_task_makespan = dist;
                    }
                }
                if (min_task_i != -1){
                    proposed_schedule[agent_id] = min_task_i;
                    free_agents.erase(agent_id);
                    free_tasks.erase(min_task_i);
                }
                //check for timeout every 10 task evaluations
                if (count % 10 == 0 && std::chrono::steady_clock::now() > endtime){
                    timeout = true;
                    break;
                }
                count++;   
            }
        }
    }

    #ifndef NDEBUG
    cout << "Time Usage: " <<  ((float)(clock() - start))/CLOCKS_PER_SEC <<endl;
    cout << "new free agents: " << env->new_freeagents.size() << " new tasks: "<< env->new_tasks.size() <<  endl;
    cout << "free agents: " << free_agents.size() << " free tasks: " << free_tasks.size() << endl;
    #endif
    return;
}
}

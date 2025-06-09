#include "schedulerActivatedGreedy.h"

namespace schedulerActivatedGreedy{

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

    int min_task_i, min_task_makespan, dist, c_loc, count, t_loc, node;
    clock_t start = clock();


    bool activation_active = false;
    std::vector<std::vector<int>> activation_action;
    if (action_dict.find("activation_action") != action_dict.end()) {
        activation_action = action_dict.at("activation_action").cast<std::vector<std::vector<int>>>();
        activation_active = true;        
    }
    


    // iterate over the free agents to decide which task to assign to each of them
    std::unordered_set<int>::iterator it = free_agents.begin();
    while (it != free_agents.end())
    {
        //keep assigning until timeout
        if (std::chrono::steady_clock::now() > endtime)
        {
            break;
        }
        int i = *it;

        assert(env->curr_task_schedule[i] == -1);

        if (activation_active){
            int agent_loc = env->curr_states.at(i).location;
            node = env->nodes->regions.at(agent_loc);
            if (activation_action[node][0] != 1){
                // if the agent is not activated, skip to the next agent
                proposed_schedule[i] = -1;
                it++;
                continue;
            }
        }
            
        min_task_i = -1;
        min_task_makespan = INT_MAX;
        count = 0;

        // iterate over all the unassigned tasks to find the one with the minimum makespan for agent i
        for (int t_id : free_tasks)
        {
            //check for timeout every 10 task evaluations
            if (count % 10 == 0 && std::chrono::steady_clock::now() > endtime)
            {
                break;
            }
            count++;

            if (activation_active){
                t_loc = env->task_pool[t_id].locations[0];
                node = env->nodes->regions.at(t_loc);
                if (activation_action[node][1] != 1){
                    continue;
                }
            }
            




            
            c_loc = env->curr_states.at(i).location;
            dist = DefaultPlanner::get_h(env, c_loc, env->task_pool[t_id].locations[0]);


            // dist = 0;
            // iterate over the locations (errands) of the task to compute the makespan to finish the task
            // makespan: the time for the agent to complete all the errands of the task t_id in order
            // for (int loc : env->task_pool[t_id].locations){
            //     dist += DefaultPlanner::get_h(env, c_loc, loc);
            //     c_loc = loc;
            // }
            

            // update the new minimum makespan
            if (dist < min_task_makespan){
                min_task_i = t_id;
                min_task_makespan = dist;
            }           
        }

        // assign the best free task to the agent i (assuming one exists)
        if (min_task_i != -1){
            proposed_schedule[i] = min_task_i;
            it = free_agents.erase(it);
            free_tasks.erase(min_task_i);
        }
        // nothing to assign
        else{
            proposed_schedule[i] = -1;
            it++;
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

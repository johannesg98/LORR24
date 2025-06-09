#include "schedulerActivatedAdvantage.h"

namespace schedulerActivatedAdvantage{

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

    int min_task_i, min_task_makespan, dist, c_loc, count;
    clock_t start = clock();

    // create dist matrix (task_id = unqiue id in task pool | task_idx = 0,1,2,... for all free tasks this turn -> used for dist_matrix)
    // std::vector<std::vector<int>> dist_matrix(env->num_of_agents, std::vector<int>(free_tasks.size(), -1));
    // std::vector<int> task_idx_to_id(free_tasks.size());
    // std::unordered_map<int, int> task_id_to_idx;
    // int task_idx = 0;
    // for (int t_id : free_tasks){
    //     task_idx_to_id[task_idx] = t_id;
    //     task_id_to_idx[t_id] = task_idx;

    //     int t_loc = env->task_pool[t_id].get_next_loc();

    //     for (int agent_id : free_agents){
    //         int agent_loc = env->curr_states.at(agent_id).location;
    //         dist = DefaultPlanner::get_h(env, agent_loc, t_loc);
    //         dist_matrix[agent_id][task_idx] = dist;
    //     }
    //     task_idx++;
    // // }

    // std::cout << "Dist matrix created." << std::endl;
    // std::vector<int> closest_agent_distance(free_tasks.size(), INT_MAX);
    // int t_loc, t_idx, agent_loc, advantage;


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


        // // get closest distance for each task to any agent apart from i
        // bool first = true;
        // for (int t_id : free_tasks) {
        //     t_idx = task_id_to_idx[t_id];
        //     for (int agent_id : free_agents) {
        //         if (agent_id != i) {
        //             dist = dist_matrix[agent_id][t_idx];
        //             // if this is the first agent or the distance is smaller than the current closest distance
        //             if (first || dist < closest_agent_distance[t_idx]) {
        //                 closest_agent_distance[t_idx] = dist;
        //                 first = false;
        //             }
        //         }
        //     }
        // }

        // // std::cout << "Agent " << i << " done." << std::endl;

        // // get advantage for each task
        // int max_advantage = INT_MIN;
        // int best_task_id = -1;
        // for (int t_id : free_tasks) {
        //     t_idx = task_id_to_idx[t_id];
        //     dist = dist_matrix[i][t_idx];
        //     advantage = closest_agent_distance[t_idx] - dist;
        //     if (advantage > max_advantage) {
        //         max_advantage = advantage;
        //         best_task_id = t_id;
        //     }
        // }

        // if (best_task_id != -1) {
        //     // assign the best task to agent i
        //     proposed_schedule[i] = best_task_id;
        //     it = free_agents.erase(it);
        //     free_tasks.erase(best_task_id);
        // } else {
        //     // no task available for agent i
        //     proposed_schedule[i] = -1;
        //     it++;
        // }




        // get closest distance for each task to any agent apart from i
        std::unordered_map<int, int> closest_agent_distance;
        for (int t_id : free_tasks) {
            int t_loc = env->task_pool[t_id].get_next_loc();
            for (int agent_id : free_agents) {
                if (agent_id != i) {
                    int agent_loc = env->curr_states.at(agent_id).location;
                    dist = DefaultPlanner::get_h(env, agent_loc, t_loc);
                    // if this is the first agent or the distance is smaller than the current closest distance
                    if (closest_agent_distance.find(t_id) == closest_agent_distance.end() || dist < closest_agent_distance[t_id]) {
                        closest_agent_distance[t_id] = dist;
                    }
                }
            }
        }

        // std::cout << "Agent " << i << " done." << std::endl;

        // get advantage for each task
        int this_agent_loc = env->curr_states.at(i).location;
        int max_advantage = INT_MIN;
        int best_task_id = -1;
        for (int t_id : free_tasks) {
            int t_loc = env->task_pool[t_id].get_next_loc();
            dist = DefaultPlanner::get_h(env, this_agent_loc, t_loc);
            int advantage = closest_agent_distance[t_id] - dist;
            if (advantage > max_advantage) {
                max_advantage = advantage;
                best_task_id = t_id;
            }
        }

        if (best_task_id != -1) {
            // assign the best task to agent i
            proposed_schedule[i] = best_task_id;
            it = free_agents.erase(it);
            free_tasks.erase(best_task_id);
        } else {
            // no task available for agent i
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

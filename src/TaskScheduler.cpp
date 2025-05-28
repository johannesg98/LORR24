#include "TaskScheduler.h"

#include "schedulerILP.h"
#include "schedulerILPsparse.h"
#include "schedulerTEMPLATE.h"
#include "schedulerPreReserve.h"
#include "schedulerRL.h"
#include "scheduler.h"
#include "const.h"
#include "heuristics.h"

//NoManSky Solution
#include "schedulerNoMan.hpp"
#include <Objects/Basic/time.hpp>
#include <Objects/Environment/environment.hpp>




/**
 * Initializes the task scheduler with a given time limit for preprocessing.
 * 
 * This function prepares the task scheduler by allocating up to half of the given preprocessing time limit 
 * and adjust for a specified tolerance to account for potential timing errors. 
 * It ensures that initialization does not exceed the allocated time.
 * 
 * @param preprocess_time_limit The total time limit allocated for preprocessing (in milliseconds).
 *
 */
void TaskScheduler::initialize(int preprocess_time_limit)
{
    solveTimeSum = 0;
    //give at most half of the entry time_limit to scheduler;
    //-SCHEDULER_TIMELIMIT_TOLERANCE for timing error tolerance
    int limit = preprocess_time_limit/2 - DefaultPlanner::SCHEDULER_TIMELIMIT_TOLERANCE;
    
    
    if (scheduler_type.empty()){
        // fallback for LRR standard build
        DefaultPlanner::schedule_initialize(limit, env);
    }
    else if (scheduler_type == "default"){
        DefaultPlanner::schedule_initialize(limit, env);
    }
    else if (scheduler_type == "ILP"){
        schedulerILP::schedule_initialize(limit, env);
    }
    else if (scheduler_type == "ILPsparse"){
        schedulerILPsparse::schedule_initialize(limit, env);
    }
    else if (scheduler_type == "RL"){
        schedulerRL::schedule_initialize(limit, env);
    }
    else if (scheduler_type == "NoManSky"){
        schedulerNoMan = MyScheduler(env);
        init_environment(*env);
    }
    else{
        std::cerr << "Unknown scheduler type: " << scheduler_type << std::endl;
        exit(1);
    }
    
    

    task_search_start_times.resize(env->num_of_agents, 0);
}

/**
 * Plans a task schedule within a specified time limit.
 * 
 * This function schedules tasks by calling shedule_plan function in default planner with half of the given time limit,
 * adjusted for timing error tolerance. The planned schedule is output to the provided schedule vector.
 * 
 * @param time_limit The total time limit allocated for scheduling (in milliseconds).
 * @param proposed_schedule A reference to a vector that will be populated with the proposed schedule (next task id for each agent).
 */

void TaskScheduler::plan(int time_limit, std::vector<int> & proposed_schedule, const std::unordered_map<std::string, pybind11::object>& action_dict)
{
    //give at most half of the entry time_limit to scheduler;
    //-SCHEDULER_TIMELIMIT_TOLERANCE for timing error tolerance
    int limit = time_limit/2 - DefaultPlanner::SCHEDULER_TIMELIMIT_TOLERANCE;
    
    std::vector<int> proposed_schedule_old = proposed_schedule;

    for (int i = 0; i < env->num_of_agents; i++){
        if (proposed_schedule[i] == -1){
            solveCount++;
            break;
        }
    }
    auto start = std::chrono::high_resolution_clock::now();




    // DefaultPlanner::reset_heuristictable(env);





    if (scheduler_type.empty()){
        // fallback for LRR standard build
        DefaultPlanner::schedule_plan(limit, proposed_schedule, env);
    }
    else if (scheduler_type == "default"){
        DefaultPlanner::schedule_plan(limit, proposed_schedule, env);
    }
    else if (scheduler_type == "ILP"){
        schedulerILP::schedule_plan(limit, proposed_schedule, env);
    }
    else if (scheduler_type == "ILPsparse"){
        schedulerILPsparse::schedule_plan(limit, proposed_schedule, env);
    }
    else if (scheduler_type == "RL"){
        schedulerRL::schedule_plan(limit, proposed_schedule, env, action_dict);
    }
    else if (scheduler_type == "NoManSky"){
        TimePoint end_time = env->plan_start_time + Milliseconds(time_limit - 10);
        update_environment(*env);
        schedulerNoMan.plan(end_time, proposed_schedule);
    }



    auto end = std::chrono::high_resolution_clock::now();
    solveTimeSum += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Scheduler average solve time per step: " << (double)solveTimeSum/ solveCount  << "ms" << std::endl;
    






    

    double Astar_reward = 0;
    double idle_agents = 0;
    int tasks_assigned = 0;
    int dist_reward = 0;
    std::vector<int> task_search_durations;
    std::vector<int> task_distances;
    
    int max_dist = (env->rows + env->cols) / 1;
    for (int agent = 0; agent < proposed_schedule_old.size(); agent++){
        // agent finished task last step and starts searching for new one this step
        if (proposed_schedule_old[agent] == -1 && task_search_start_times[agent] == -1){
            task_search_start_times[agent] = env->curr_timestep;
          
        }
        // agent received a new task
        if (proposed_schedule_old[agent] == -1 && proposed_schedule[agent] != -1){
            int dist = DefaultPlanner::get_h(env, env->curr_states[agent].location, env->task_pool[proposed_schedule[agent]].locations[0]);
            task_distances.push_back(dist);
            dist_reward += -dist;
            double rew = max_dist - dist;
            rew = static_cast<float>(rew) / max_dist;
            // float sign = 0;
            // if (rew != 0){
            //     sign = rew / abs(rew);
            // }
            rew = rew*rew*rew*rew;    // rew^4, otherwise often high rewards
            Astar_reward += rew;//(rew-0.5)*20; // onyl rew
            tasks_assigned++;
            task_search_durations.push_back(env->curr_timestep - task_search_start_times[agent]);
            task_search_start_times[agent] = -1;            
        }
        // agent is idle and did not get a task
        if (proposed_schedule[agent] == -1){
            idle_agents++;
        }
    }
    
    env->Astar_reward = Astar_reward; //*20
    env->idle_agents_reward = -idle_agents; // *20
    env->tasks_assigned_reward = tasks_assigned;
    env->task_search_durations = task_search_durations;
    env->dist_reward = dist_reward;
    env->task_distances = task_distances;


    //Backtracking
    BacktrackBundle backtrack_bundle(env->curr_timestep);
    env->backtrack_times_whole_task.clear();
    for (int agent = 0; agent < proposed_schedule_old.size(); agent++){
        // agent finished task last step and starts searching for new one this step
        if(task_search_start_times[agent] == env->curr_timestep){
            for (auto bundle_it = env->backtrack_bundles_whole_task.begin(); bundle_it != env->backtrack_bundles_whole_task.end(); ++bundle_it) {
                if (bundle_it->agents_unfinished.find(agent) != bundle_it->agents_unfinished.end()) {
                    bundle_it->traveltimes.push_back(env->curr_timestep - bundle_it->start_time);
                    bundle_it->agents_unfinished.erase(agent);
                    if (bundle_it->agents_unfinished.empty()){
                        env->backtrack_times_whole_task[bundle_it->start_time] = bundle_it->traveltimes;
                        env->backtrack_bundles_whole_task.erase(bundle_it);
                    }
                    break;
                }
            }
        }
        // agent received a new task
        if (proposed_schedule_old[agent] == -1 && proposed_schedule[agent] != -1){
            backtrack_bundle.agents_unfinished.insert(agent);
        }
    }
    
    env->backtrack_rewards_first_errand.clear();
    env->backtrack_rewards_whole_task.clear();

    for (const auto& [starttime, timesVec] : env->backtrack_times_first_errand){
        if (timesVec[0] == -1)
            continue;
        double rew_sum = 0;
        for (int time : timesVec){
            double rew;
            if (backtrack_reward_type == "DividedTime"){
                // if (time == 0)
                //     rew = 1;
                // else
                rew = (double)1.0/(time+20);
            }
            else if (backtrack_reward_type == "MaxDist-Time"){
                rew = max_dist - time;
            }
            else if (backtrack_reward_type == "SquaredNormalized"){
                rew = max_dist - time;
                rew = rew / max_dist;
                // rew = (rew - 0.5) * 2;
                float sign = 0;
                if (rew != 0){
                    sign = rew / abs(rew);
                }
                rew = sign * rew*rew*rew*rew;
            }
            
            rew_sum += rew;
        }
        env->backtrack_rewards_first_errand[starttime] = rew_sum;
    }
    env->backtrack_times_first_errand.clear();

    if (backtrack_bundle.agents_unfinished.size() > 0){
        env->backtrack_bundles_first_errand.push_back(backtrack_bundle);
        env->backtrack_bundles_whole_task.push_back(backtrack_bundle);
    }
    else{
        env->backtrack_times_first_errand[backtrack_bundle.start_time] = std::vector<int>{-1};
        env->backtrack_times_whole_task[backtrack_bundle.start_time] = std::vector<int>{-1};
        env->backtrack_rewards_first_errand[backtrack_bundle.start_time] = 0;
        env->backtrack_rewards_whole_task[backtrack_bundle.start_time] = 0;
    }
    
    for (const auto& [starttime, timesVec] : env->backtrack_times_whole_task){
        if (timesVec[0] == -1)
            continue;
        double rew_sum = 0;
        for (int time : timesVec){
            double rew = max_dist - time;
            rew = rew / max_dist;
            float sign = 0;
            if (rew != 0){
                sign = rew / abs(rew);
            }
            rew = sign * rew*rew*rew*rew;
            // rew = rew;
            rew_sum += rew;
        }
        env->backtrack_rewards_whole_task[starttime] = rew_sum;
    }
}

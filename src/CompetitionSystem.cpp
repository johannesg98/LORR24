#include <cmath>
#include "CompetitionSystem.h"
#include <boost/tokenizer.hpp>
#include "SharedEnv.h"
#include "nlohmann/json.hpp"
#include <functional>
#include <Logger.h>
#include <heuristics.h>
#include "planner.h"  // Add this to access DefaultPlanner functions


using json = nlohmann::ordered_json;


//NoManSky Solution
#include "schedulerNoMan.hpp"
#include <Objects/Basic/time.hpp>
#include <Objects/Environment/environment.hpp>

//RL stuff
#include <Roadmap.h>






// // This function might not work correctly with small map (w or h <=2)
// bool BaseSystem::valid_moves(vector<State>& prev, vector<Action>& action)
// {
//   return model->is_valid(prev, action);
// }


void BaseSystem::sync_shared_env() 
{
    if (!started)
    {
        env->goal_locations.resize(num_of_agents);
        task_manager.sync_shared_env(env);
        simulator.sync_shared_env(env);

        if (simulator.get_curr_timestep() == 0)
        {
            env->new_freeagents.reserve(num_of_agents); //new free agents are empty in task_manager on initialization, set it after task_manager sync
            for (int i = 0; i < num_of_agents; i++)
            {
                env->new_freeagents.push_back(i);
            }
        }
        //update proposed action to all wait
        proposed_actions.clear();
        proposed_actions.resize(num_of_agents, Action::W);
        //update proposed schedule to previous assignment
        proposed_schedule = env->curr_task_schedule;
        
    }
    else
    {
        env->curr_timestep = simulator.get_curr_timestep();
    }
}


bool BaseSystem::planner_wrapper()
{
    planner->compute(plan_time_limit, proposed_actions, proposed_schedule);
    return true;
}


void BaseSystem::plan(int & timeout_timesteps)
{

    using namespace std::placeholders;
    int timestep = simulator.get_curr_timestep();

    std::packaged_task<bool()> task(std::bind(&BaseSystem::planner_wrapper, this));


    future = task.get_future();
    if (task_td.joinable()){
        task_td.join();
    }
    env->plan_start_time = std::chrono::steady_clock::now();
    task_td = std::thread(std::move(task));

    started = true;

    while (timestep + timeout_timesteps < simulation_time){

        if (future.wait_for(std::chrono::milliseconds(plan_time_limit)) == std::future_status::ready)
            {
                task_td.join();
                started = false;
                auto res = future.get();

                logger->log_info("planner returns", timestep + timeout_timesteps);
                return;
            }
        logger->log_info("planner timeout", timestep + timeout_timesteps);
        timeout_timesteps += 1;
    }

    //
}


bool BaseSystem::planner_initialize()
{
    using namespace std::placeholders;
    std::packaged_task<void(int)> init_task(std::bind(&Entry::initialize, planner, _1));
    auto init_future = init_task.get_future();
    
    env->plan_start_time = std::chrono::steady_clock::now();
    auto init_td = std::thread(std::move(init_task), preprocess_time_limit);
    if (init_future.wait_for(std::chrono::milliseconds(preprocess_time_limit)) == std::future_status::ready)
    {
        init_td.join();
        return true;
    }

    init_td.detach();
    return false;
}


void BaseSystem::log_preprocessing(bool succ)
{
    if (logger == nullptr)
        return;
    if (succ)
    {
        logger->log_info("Preprocessing success", simulator.get_curr_timestep());
    } 
    else
    {
        logger->log_fatal("Preprocessing timeout", simulator.get_curr_timestep());
    }
    logger->flush();
}


void BaseSystem::simulate(int simulation_time)
{
    //init logger
    //Logger* log = new Logger();
    initialize();

    this->simulation_time = simulation_time;

    vector<Action> all_wait_actions(num_of_agents, Action::NA);

    for (; simulator.get_curr_timestep() < simulation_time; )
    {
        // find a plan
        sync_shared_env();

        auto start = std::chrono::steady_clock::now();

        int timeout_timesteps = 0;

        plan(timeout_timesteps);

        auto end = std::chrono::steady_clock::now();

        for (int i = 0 ; i< timeout_timesteps; i ++){
            simulator.move(all_wait_actions);
            for (int a = 0; a < num_of_agents; a++)
                {
                    if (!env->goal_locations[a].empty())
                        solution_costs[a]++;
                }
        }

        total_timetous+=timeout_timesteps;

        if (simulator.get_curr_timestep() >= simulation_time){

            auto diff = end-start;
            planner_times.push_back(std::chrono::duration<double>(diff).count());
            break;
        }

        for (int a = 0; a < num_of_agents; a++)
        {
            if (!env->goal_locations[a].empty())
                solution_costs[a]++;
        }

        // move drives
        vector<State> curr_states = simulator.move(proposed_actions);
        int timestep = simulator.get_curr_timestep();
        // agents do not move


        auto diff = end-start;
        planner_times.push_back(std::chrono::duration<double>(diff).count());

        // update tasks
        task_manager.update_tasks(curr_states, proposed_schedule, simulator.get_curr_timestep());
    }
}


void BaseSystem::initialize()
{
    env->num_of_agents = num_of_agents;
    env->rows = map.rows;
    env->cols = map.cols;
    env->map = map.map;

    
    // // bool succ = load_records(); // continue simulating from the records
    // timestep = 0;
    // curr_states = starts;

    int timestep = simulator.get_curr_timestep();

    //planner initilise before knowing the first goals
    bool planner_initialize_success= planner_initialize();
    
    log_preprocessing(planner_initialize_success);
    if (!planner_initialize_success)
        _exit(124);

    // initialize_goal_locations();

    task_manager.reveal_tasks(timestep); //this also intialize env->new_tasks

    sync_shared_env();


    solution_costs.resize(num_of_agents);
    for (int a = 0; a < num_of_agents; a++)
    {
        solution_costs[a] = 0;
    }

    proposed_actions.resize(num_of_agents, Action::W);
    proposed_schedule.resize(num_of_agents, -1);
}


void BaseSystem::saveResults(const string &fileName, int screen) const
{
    json js;
    // Save action model
    js["actionModel"] = "MAPF_T";
    js["version"] = "2024 LoRR";

    // std::string feasible = fast_mover_feasible ? "Yes" : "No";
    // js["AllValid"] = feasible;

    js["teamSize"] = num_of_agents;

    js["numTaskFinished"] = task_manager.num_of_task_finish;
    int makespan = 0;
    if (num_of_agents > 0)
    {
        makespan = solution_costs[0];
        for (int a = 1; a < num_of_agents; a++)
        {
            if (solution_costs[a] > makespan)
            {
                makespan = solution_costs[a];
            }
        }
    }
    js["makespan"] = makespan;

    js["numPlannerErrors"] = simulator.get_number_errors();
    js["numScheduleErrors"] = task_manager.get_number_errors();

    js["numEntryTimeouts"] = total_timetous;

    // Save start locations[x,y,orientation]
    if (screen <= 2)
    {
        js["start"] = simulator.starts_to_json();
    }
    
    if (screen <= 2)
    {
        js["actualPaths"] = simulator.actual_path_to_json();
    }

    if (screen <=1)
    {
        js["plannerPaths"] = simulator.planned_path_to_json();

        json planning_times = json::array();
        for (double time: planner_times)
            planning_times.push_back(time);
        js["plannerTimes"] = planning_times;

        // Save errors
        js["errors"] = simulator.action_errors_to_json();

        //actual schedules
        json aschedules = json::array();
        for (int i = 0; i < num_of_agents; i++)
        {
            std::string schedules;
            bool first = true;
            for (const auto schedule : task_manager.actual_schedule[i])
            {
                if (!first)
                {
                    schedules+= ",";
                } 
                else 
                {
                    first = false;
                }

                schedules+=std::to_string(schedule.first);
                schedules+=":";
                int tid = schedule.second;
                schedules+=std::to_string(tid);
            }  
            aschedules.push_back(schedules);
        }

        js["actualSchedule"] = aschedules;

        //planned schedules
        json pschedules = json::array();
        for (int i = 0; i < num_of_agents; i++)
        {
            std::string schedules;
            bool first = true;
            for (const auto schedule : task_manager.planner_schedule[i])
            {
                if (!first)
                {
                    schedules+= ",";
                } 
                else 
                {
                    first = false;
                }

                schedules+=std::to_string(schedule.first);
                schedules+=":";
                int tid = schedule.second;
                schedules+=std::to_string(tid);
                
            }  
            pschedules.push_back(schedules);
        }

        js["plannerSchedule"] = pschedules;

        // Save errors
        json schedule_errors = json::array();
        for (auto error: task_manager.schedule_errors)
        {
            std::string error_msg;
            int t_id;
            int agent1;
            int agent2;
            int timestep;
            std::tie(error_msg,t_id,agent1,agent2,timestep) = error;
            json e = json::array();
            e.push_back(t_id);
            e.push_back(agent1);
            e.push_back(agent2);
            e.push_back(timestep);
            e.push_back(error_msg);
            schedule_errors.push_back(e);
        }

        js["scheduleErrors"] = schedule_errors;

        // Save events
        json event = json::array();
        for(auto e: task_manager.events)
        {
            json ev = json::array();
            int timestep;
            int agent_id;
            int task_id;
            int seq_id;
            std::tie(timestep,agent_id,task_id,seq_id) = e;
            ev.push_back(timestep);
            ev.push_back(agent_id);
            ev.push_back(task_id);
            ev.push_back(seq_id);
            event.push_back(ev);
        }
        js["events"] = event;

        // Save all tasks
        json tasks = task_manager.to_json(map.cols);
        js["tasks"] = tasks;

        // Save nodes and action_rl
        if (env->nodes && !env->action_rl.empty())
        {
            js["nodeRegions"] = simulator.node_regions_to_json(env);
            js["action_rl"] = simulator.action_rl_to_json(env);
        }

    }

    std::ofstream f(fileName,std::ios_base::trunc |std::ios_base::out);
    f << std::setw(4) << js;

}


void BaseSystem::initializeExtendedBaseSystem(int simulation_time) {
    initialize();
    this->simulation_time = simulation_time;
    sync_shared_env();
}

pybind11::dict BaseSystem::get_NoManSkySolution(int time_limit) {
    if (env->curr_timestep == 0){
        schedulerNoMan = MyScheduler(env);
        init_environment(*env);
        std::cout << "NoManSky scheduler initialized" << std::endl;
    }

    env->plan_start_time = std::chrono::steady_clock::now();

    TimePoint end_time = env->plan_start_time + Milliseconds(time_limit);
    update_environment(*env);
    std::vector<int> proposed_schedule_copy = proposed_schedule;
    
    schedulerNoMan.plan(end_time, proposed_schedule_copy);

    std::vector<double> distribution(env->nodes->nNodes, 0);
    std::vector<std::vector<bool>> activation(env->nodes->nNodes, std::vector<bool>(2, false));

    int sum = 0;
    for (int agent=0; agent < env->num_of_agents; agent++){
        if (proposed_schedule[agent] == -1 && proposed_schedule_copy[agent] != -1){
            int task_loc = env->task_pool[proposed_schedule_copy[agent]].locations[0];
            int node = env->nodes->regions[task_loc];
            distribution[node]++;
            sum++;

            activation[node][1] = true;
            int agent_loc = env->curr_states[agent].location;
            int agent_node = env->nodes->regions[agent_loc];
            activation[agent_node][0] = true;
        }
    }
    
    
    if (sum > 0){
        for (int i=0; i<env->nodes->nNodes; i++){
            distribution[i] = distribution[i]/sum;
        }
    }

    pybind11::dict solution_dict;
    solution_dict["distribution"] = distribution;
    solution_dict["activation"] = activation;


    return solution_dict;
}

bool BaseSystem::step(const std::unordered_map<std::string, pybind11::object>& action_dict){
    bool done = false;

    auto start = std::chrono::steady_clock::now();

    env->plan_start_time = std::chrono::steady_clock::now();

    planner->compute(plan_time_limit, proposed_actions, proposed_schedule, action_dict);

    auto end = std::chrono::steady_clock::now();
    auto diff = end-start;
    planner_times.push_back(std::chrono::duration<double>(diff).count());

    for (int a = 0; a < num_of_agents; a++){
        if (!env->goal_locations[a].empty())
            solution_costs[a]++;
    }

    vector<State> curr_states = simulator.move(proposed_actions);

    get_roadmap_reward(curr_states);

    task_manager.update_tasks(curr_states, proposed_schedule, simulator.get_curr_timestep());

    logger->log_info("Step done.", simulator.get_curr_timestep());

    sync_shared_env();

    if (simulator.get_curr_timestep() == simulation_time){
        done = true;
    }

    return done;
}

void BaseSystem::get_roadmap_reward(const std::vector<State>& curr_states) {
    if (env->roadmap != nullptr) {
        // only during first time step since we dont have previous goals
        if (env->curr_timestep == 0){
            std::cout << "Roadmap reward initialized during first step." << std::endl;
            int this_distance_sum = 0;
            for (int agent=0; agent < env->num_of_agents; agent++){
                if (!env->goal_locations[agent].empty()){
                    int goal_loc = env->goal_locations[agent][0].first;
                    int agent_loc = env->curr_states[agent].location;
                    this_distance_sum += DefaultPlanner::get_h(env, agent_loc, goal_loc);
                    last_agent_goals.push_back(goal_loc);
                } else {
                    last_agent_goals.push_back(-1);
                }                
            }
            roadmap_progress_reward = env->roadmap_reward_first_distance - this_distance_sum;
            last_roadmap_distance_sum = this_distance_sum;
        }
        // all other time steps
        else {
            int this_distance_sum = 0;
            int ignored_distance_sum = 0;
            for (int agent=0; agent < env->num_of_agents; agent++){
                if (!env->goal_locations[agent].empty()){
                    int goal_loc = env->goal_locations[agent][0].first;
                    if (goal_loc == last_agent_goals[agent]){
                        this_distance_sum += DefaultPlanner::get_h(env, curr_states[agent].location, goal_loc);
                    }
                    else {
                        ignored_distance_sum += DefaultPlanner::get_h(env, curr_states[agent].location, goal_loc);
                        last_agent_goals[agent] = goal_loc;
                    }
                }
                else {
                    last_agent_goals[agent] = -1;
                }
            }
            roadmap_progress_reward = last_roadmap_distance_sum - this_distance_sum;
            last_roadmap_distance_sum = this_distance_sum + ignored_distance_sum;
        }
    }
}

pybind11::dict BaseSystem::get_reward(){
    pybind11::dict reward_dict;

    reward_dict["task-finished"] = task_manager.num_of_task_finish - num_of_task_finish_last_call;
    num_of_task_finish_last_call = task_manager.num_of_task_finish;

    reward_dict["first-errands-started"] = task_manager.num_of_first_errands_started - num_of_first_errands_started_last_call;
    num_of_first_errands_started_last_call = task_manager.num_of_first_errands_started;
  
    reward_dict["A*-distance"] = env->Astar_reward;

    reward_dict["idle-agents"] = env->idle_agents_reward;

    reward_dict["tasks-assigned"] = env->tasks_assigned_reward;

    reward_dict["dist-reward"] = env->dist_reward;

    reward_dict["backtrack-rewards-first-errand"] = env->backtrack_rewards_first_errand;

    reward_dict["backtrack-rewards-whole-task"] = env->backtrack_rewards_whole_task;   
    
    reward_dict["CTBT-rewards"] = env->CTBT_rewards;

    // Roadmap reward
    if (env->roadmap != nullptr){
        reward_dict["roadmap-progress-reward"] = roadmap_progress_reward;
    }
    
    
    return reward_dict;
}

pybind11::dict BaseSystem::get_info(){
    pybind11::dict info_dict;

    info_dict["task-search-durations"] = env->task_search_durations;

    info_dict["task-distances"] = env->task_distances;

    info_dict["backtrack-times-first-errand"] = env->backtrack_times_first_errand;

    info_dict["backtrack-times-whole-task"] = env->backtrack_times_whole_task;
    
    return info_dict;
}

int BaseSystem::loadNodes(const std::string& fname){
    env->nodes = std::make_unique<Nodes>(fname);
    return env->nodes->nNodes;
}

int BaseSystem::loadRoadmapNodes(const std::string& fname) {
    roadmap = Roadmap(env, fname);
    env->roadmap = &roadmap;
    return env->roadmap->graph.nNodes;
}



pybind11::dict BaseSystem::get_observation(std::unordered_set<std::string>& observationTypes){
    
    pybind11::dict obs;
    clock_t start = clock();

    // time
    obs["time"] = env->curr_timestep;

    //////////////////////////////////////////////////////////////////
    ////////////////////// Basic Node Observations ////////////////////////

    std::vector<int> agents_per_node;
    std::vector<int> free_agents_per_node;
    if (observationTypes.count("node-basics")){
        
        

        // agents per node
        // free agents per node
        agents_per_node.resize(env->nodes->nNodes, 0);
        free_agents_per_node.resize(env->nodes->nNodes, 0);
        for (int i=0; i < env->num_of_agents; i++){
            int loc = env->curr_states[i].location;
            int node = env->nodes->regions[loc];
            agents_per_node[node]++;
            if (env->curr_task_schedule[i] == -1){
                free_agents_per_node[node]++;
            }
        }
        obs["agents_per_node"] = agents_per_node;
        obs["free_agents_per_node"] = free_agents_per_node;

        // free tasks per node
        std::vector<int> free_tasks_per_node(env->nodes->nNodes, 0);
        for (const auto& [id, task] : env->task_pool){
            if (task.agent_assigned == -1){
                int node = env->nodes->regions[task.locations[0]];
                free_tasks_per_node[node]++;
            }
        }
        obs["free_tasks_per_node"] = free_tasks_per_node;

        // in case we allow task changes: we add already assigned tasks and agents that didnt arrive at the first errand yet back into free agents and tasks
        if (env->allow_task_change){
            for (int i=0; i < env->num_of_agents; i++){
                int task_id = env->curr_task_schedule[i];
                if (task_id != -1 and env->task_pool[task_id].idx_next_loc == 0){
                    int a_loc = env->curr_states[i].location;
                    int a_node = env->nodes->regions[a_loc];
                    free_agents_per_node[a_node]++;
                    int t_loc = env->task_pool[task_id].locations[0];
                    int t_node = env->nodes->regions[t_loc];
                    free_tasks_per_node[t_node]++;
                }
            }
            obs["free_agents_per_node"] = free_agents_per_node;
            obs["free_tasks_per_node"] = free_tasks_per_node;
        }
    }




    //////////////////////////////////////////////////////////////////
    ////////////// Advanced Node (+edge) Observations ////////////////

    if (observationTypes.count("node-advanced")){
        if (!observationTypes.count("node-basics")){
            throw std::runtime_error("node-advanced observation requires node-basics observation");
        }

        // distance until first agent becomes available at node (one scalar entry)
        std::vector<int> distance_until_agent_available(env->nodes->nNodes, distance_until_agent_avail_MAX);
        for (int agent=0; agent<env->num_of_agents; agent++){
            int agent_loc = env->curr_states[agent].location;
            int task_id = env->curr_task_schedule[agent];
            if (task_id == -1 or (env->allow_task_change and env->task_pool[task_id].idx_next_loc == 0)){
                int node = env->nodes->regions[agent_loc];
                distance_until_agent_available[node] = 0;
            }
            else if (env->task_pool[task_id].idx_next_loc == env->task_pool[task_id].locations.size()-1){
                int task_loc = env->task_pool[task_id].locations.back();
                int distance = DefaultPlanner::get_h(env, agent_loc, task_loc);
                int node = env->nodes->regions[task_loc];
                distance_until_agent_available[node] = std::min(distance_until_agent_available[node], distance);
            }
        }
        obs["distance_until_agent_available_per_node"] = distance_until_agent_available;

        // distance until agent becomes available at node (individual entry for each next step)
        std::vector<std::vector<int>> agents_available_next_steps(env->nodes->nNodes, std::vector<int>(distance_until_agent_avail_MAX, 0));
        for (int agent=0; agent<env->num_of_agents; agent++){
            int agent_loc = env->curr_states[agent].location;
            int task_id = env->curr_task_schedule[agent];
            if (task_id == -1 or (env->allow_task_change and env->task_pool[task_id].idx_next_loc == 0)){
                int node = env->nodes->regions[agent_loc];
                agents_available_next_steps[node][0]++;
            }
            else if (env->task_pool[task_id].idx_next_loc == env->task_pool[task_id].locations.size()-1){
                int task_loc = env->task_pool[task_id].locations.back();
                int distance = DefaultPlanner::get_h(env, agent_loc, task_loc);
                if (distance < distance_until_agent_avail_MAX){
                    int node = env->nodes->regions[task_loc];
                    agents_available_next_steps[node][distance]++;
                }
            }
        }
        obs["agents_available_next_steps_per_node"] = agents_available_next_steps;


        // min task length per node
        int max_length = 2 * (env->cols + env->rows);
        std::vector<int> min_task_length_per_node(env->nodes->nNodes, max_length);
        for (auto& [id, task] : env->task_pool){
            if (task.agent_assigned == -1 or (env->allow_task_change and task.idx_next_loc == 0)){
                if (task.length == -1){
                    task.length = 0;
                    for (int i=0; i<task.locations.size()-1; i++){
                        task.length += DefaultPlanner::get_h(env, task.locations[i], task.locations[i+1]);
                    }
                }
                int node = env->nodes->regions[task.locations[0]];
                if (task.length < min_task_length_per_node[node]){
                    min_task_length_per_node[node] = task.length;
                }
            }
        }
        obs["min_task_length_per_node"] = min_task_length_per_node;



        // congestion ratio at nodes
        std::vector<float> congestion_ratio_per_node(env->nodes->nNodes, 0);
        for (int i=0; i<env->nodes->nNodes; i++){
            if (space_per_node[i] > 0){
                congestion_ratio_per_node[i] = (float)agents_per_node[i]/(float)space_per_node[i];
            }
        }
        obs["congestion_ratio_per_node"] = congestion_ratio_per_node;
        
        

        // congestion at edges (directed)
        for (int loc=0; loc<env->map.size(); loc++){
            for (int i=0; i<MP_loc_to_edges[loc].size(); i++){
                int edge_id = MP_loc_to_edges[loc][i].first;
                if (edge_id == 128){
                    int x = loc % env->cols;
                    int y = loc / env->cols;
                }
            }
        }




        std::vector<std::vector<double>> MP_congestion_per_edge(MP_edge_lengths.size(), std::vector<double>(3,0));
        for (int agent=0; agent<env->num_of_agents; agent++){
            int loc = env->curr_states[agent].location;
            for (int i=0; i<MP_loc_to_edges[loc].size(); i++){
                int edge_id = MP_loc_to_edges[loc][i].first;
                edgeFeatures::Direction edge_dir = MP_loc_to_edges[loc][i].second;
                // same direction
                if (edge_dir == env->curr_states[agent].orientation){
                    MP_congestion_per_edge[edge_id][0]++;
                }
                // opposite direction
                else if (abs(edge_dir - env->curr_states[agent].orientation) == 2){
                    MP_congestion_per_edge[edge_id][1]++;
                }
                // perpendicular direction
                else {
                    MP_congestion_per_edge[edge_id][2]++;
                }            
            }
        }
        // normalize
        for (int edge_id=0; edge_id<MP_congestion_per_edge.size(); edge_id++){
            for (int i=0; i<3; i++){
                if (MP_congestion_per_edge[edge_id][i] > 0){
                    MP_congestion_per_edge[edge_id][i] = MP_congestion_per_edge[edge_id][i]/(double)MP_edge_lengths[edge_id];
                }
            }
        }
        obs["congestion_ratio_per_edge"] = MP_congestion_per_edge;


        // congestion ratio (undirected) per edge next steps
        std::vector<std::vector<double>> congestion_ratio_next_steps_per_edge(MP_edge_lengths.size(), std::vector<double>(distance_until_agent_avail_MAX, 0));
        DefaultPlanner::TrajLNS& traj_lns_ref = DefaultPlanner::get_trajLNS();
        int traj_length;
        for (int agent=0; agent<env->num_of_agents; agent++){
            traj_length = traj_lns_ref.trajs[agent].size();

            traj_length = std::min(traj_length, distance_until_agent_avail_MAX);
            for (int step=0; step<traj_length; step++){
                int loc = traj_lns_ref.trajs[agent][step];

                for (int i=0; i<MP_loc_to_edges[loc].size(); i++){
                    int edge_id = MP_loc_to_edges[loc][i].first;
                    congestion_ratio_next_steps_per_edge[edge_id][step] ++;          
                }
            }
        }
        // normalize
        for (int edge_id=0; edge_id<congestion_ratio_next_steps_per_edge.size(); edge_id++){
            if (MP_edge_lengths[edge_id] > 0){
                for (int step=0; step<distance_until_agent_avail_MAX; step++){
                    congestion_ratio_next_steps_per_edge[edge_id][step] = congestion_ratio_next_steps_per_edge[edge_id][step]/(double)MP_edge_lengths[edge_id];
                }
            }
        }
        obs["congestion_ratio_next_steps_per_edge"] = congestion_ratio_next_steps_per_edge;
        


        // agents tell closest task to nodes
        std::vector<int> contains_closest_task_per_node(env->nodes->nNodes, 0);
        std::vector<int> closest_task_connection_per_MP_edge(MP_edge_lengths.size(), 0);
        for (int agent=0; agent<env->num_of_agents; agent++){
            int task_id = env->curr_task_schedule[agent];
            if (task_id == -1 or (env->allow_task_change and env->task_pool[task_id].idx_next_loc == 0)){
                int agent_loc = env->curr_states[agent].location;
                int closest_dist = INT_MAX;
                int closest_task_id = -1;
                for (auto& pair : env->task_pool) {
                    int task_id = pair.first;
                    Task& task = pair.second;
                    if (task.agent_assigned == -1){
                        int task_loc = task.get_next_loc();
                        int distance = DefaultPlanner::get_h(env, agent_loc, task_loc);
                        if (distance < closest_dist){
                            closest_dist = distance;
                            closest_task_id = task_id;
                        }
                    }
                }
                if (closest_task_id != -1){
                    int goal_node = env->nodes->regions[env->task_pool[closest_task_id].locations[0]];
                    contains_closest_task_per_node[goal_node] += 1;
                    int agent_node = env->nodes->regions[agent_loc];
                    if (MP_edge_map.find({agent_node, goal_node}) != MP_edge_map.end()){
                        int edge_id = MP_edge_map[{agent_node, goal_node}];
                        closest_task_connection_per_MP_edge[edge_id] = 1;
                    }
                }
            }
        }
        obs["contains_closest_task_per_node"] = contains_closest_task_per_node;
        obs["closest_task_connection_per_MP_edge"] = closest_task_connection_per_MP_edge;
        
        // agents that wait per edge
        std::vector<double> agents_waiting_per_edge(MP_edge_lengths.size(), 0);
        if (env->curr_timestep > 0){
            for (int agent=0; agent<env->num_of_agents; agent++){
                if (env->curr_states[agent].location == last_agent_states[agent].location && env->curr_states[agent].orientation == last_agent_states[agent].orientation){
                    int loc = env->curr_states[agent].location;
                    for (int i=0; i<MP_loc_to_edges[loc].size(); i++){
                        int edge_id = MP_loc_to_edges[loc][i].first;
                        agents_waiting_per_edge[edge_id]++;
                    }
                }
            }
        }
        last_agent_states = env->curr_states;
        // normalize
        for (int edge_id=0; edge_id<MP_congestion_per_edge.size(); edge_id++){
            if (MP_edge_lengths[edge_id] > 0){
                agents_waiting_per_edge[edge_id] = agents_waiting_per_edge[edge_id]/(double)MP_edge_lengths[edge_id];
            }
        }
        obs["agents_waiting_per_edge"] = agents_waiting_per_edge;

 
    }

    //////////////////////////////////////////////////////////
    ///////////////// Roadmap (RM) Observations ///////////////////



    if (observationTypes.count("roadmap-activation")){
        // agents and directions and free agents per node 

        std::vector<int> RM_agents_per_node(env->roadmap->graph.nNodes, 0);
        std::vector<int> RM_free_agents_per_node(env->roadmap->graph.nNodes, 0);
        std::vector<int> RM_agents_per_node_in_dir(env->roadmap->graph.nNodes, 0);
        std::vector<int> RM_agents_per_node_90_dir(env->roadmap->graph.nNodes, 0);
        std::vector<int> RM_agents_per_node_op_dir(env->roadmap->graph.nNodes, 0);
        for (int agent=0; agent<env->num_of_agents; agent++){
            int loc = env->curr_states[agent].location;
            int node = env->roadmap->graph.loc_to_node[loc];
            if (node != -1){
                RM_agents_per_node[node]++;
                if (env->curr_task_schedule[agent] == -1){
                    RM_free_agents_per_node[node]++;
                }
                int node_dir = env->roadmap->graph.node_direction[node];
                int agent_dir = env->curr_states[agent].orientation;
                if (node_dir == agent_dir){
                    RM_agents_per_node_in_dir[node]++;
                }
                else if (abs(node_dir - agent_dir) == 2){
                    RM_agents_per_node_op_dir[node]++;
                }
                else {
                    RM_agents_per_node_90_dir[node]++;
                }
            }
        }
        obs["RM_agents_per_node"] = RM_agents_per_node;
        obs["RM_free_agents_per_node"] = RM_free_agents_per_node;
        obs["RM_agents_per_node_in_dir"] = RM_agents_per_node_in_dir;
        obs["RM_agents_per_node_90_dir"] = RM_agents_per_node_90_dir;
        obs["RM_agents_per_node_op_dir"] = RM_agents_per_node_op_dir;

        // agents in direction per edge
        std::vector<int> RM_agents_per_edge_in_dir(env->roadmap->graph.nEdges, 0);
        std::vector<int> RM_agents_per_edge_90_dir(env->roadmap->graph.nEdges, 0);
        std::vector<int> RM_agents_per_edge_op_dir(env->roadmap->graph.nEdges, 0);
        for (int agent=0; agent<env->num_of_agents; agent++){
            int agent_loc = env->curr_states[agent].location;
            if (!env->roadmap->graph.loc_to_edges[agent_loc].empty()){
                int agent_dir = env->curr_states[agent].orientation;
                for (int edge_id : env->roadmap->graph.loc_to_edges[agent_loc]){
                    int target_node = env->roadmap->graph.edge_to_node_start_end[edge_id].second;
                    int node_dir = env->roadmap->graph.node_direction[target_node];
                    if (node_dir == agent_dir){
                        RM_agents_per_edge_in_dir[edge_id]++;
                    }
                    else if (abs(node_dir - agent_dir) == 2){
                        RM_agents_per_edge_op_dir[edge_id]++;
                    }
                    else {
                        RM_agents_per_edge_90_dir[edge_id]++;
                    }
                }
            }
        }
        obs["RM_agents_per_edge_in_dir"] = RM_agents_per_edge_in_dir;
        obs["RM_agents_per_edge_90_dir"] = RM_agents_per_edge_90_dir;
        obs["RM_agents_per_edge_op_dir"] = RM_agents_per_edge_op_dir;




    }

    std::cout << "Time for getting observations: " << (double)(clock()-start)/CLOCKS_PER_SEC << std::endl;
    return obs;
}

    //////////////// End Observations ////////////////////////
    //////////////////////////////////////////////////////////









std::tuple<int,
            int,
            std::vector<std::vector<int>>,
            std::vector<std::vector<int>>,
            std::vector<std::vector<int>>,
            std::vector<double>,
            std::vector<std::vector<double>>,
            std::vector<std::vector<std::pair<int, edgeFeatures::Direction>>>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>,
            std::vector<std::vector<double>>
                                                                                    > BaseSystem::get_env_vals(std::unordered_set<std::string>& observationTypes, int MP_edge_limit){

    int max_num_agents = env->num_of_agents;
    int max_num_tasks = env->task_pool.size();

    std::vector<std::vector<int>> AdjacencyMatrix;
    std::vector<std::vector<int>> NodeCostMatrix;
    std::vector<int> space_per_node_new;
    std::vector<std::vector<int>> MP_edge_index(2);
    std::vector<double> MP_edge_weights;
    std::vector<std::vector<double>> node_positions;
    std::vector<std::vector<std::pair<int, edgeFeatures::Direction>>> MP_loc_to_edges_new;
    std::vector<int> MP_edge_lengths_new;

    if (observationTypes.count("node-basics")) {
        AdjacencyMatrix = env->nodes->AdjacencyMatrix;


        // rebalancing optimizer cost matrix
        NodeCostMatrix.resize(env->nodes->nNodes,std::vector<int>(env->nodes->nNodes,0));
        for (int i=0; i<env->nodes->nNodes; i++){
            int start_loc = env->nodes->locations[i];
            for (int j=0; j<env->nodes->nNodes; j++){
                int target_loc = env->nodes->locations[j];
                NodeCostMatrix[i][j] = DefaultPlanner::get_h(env, start_loc, target_loc);
            }
        }

        // count free spaces per node to calc congestion ratio
        space_per_node_new.resize(env->nodes->nNodes, 0);
        for (int id=0; id<env->map.size(); id++){
            int node = env->nodes->regions[id];
            if (env->map[id] == 0){
                space_per_node_new[node]++;
            }
        }


        // message passing
        clock_t start = clock();
        MP_loc_to_edges_new.resize(env->map.size());
        if (MP_edge_limit > 0){
            
            for (int o=0; o < env->nodes->nNodes; o++){
                for (int d=0; d < env->nodes->nNodes; d++){
                    if (NodeCostMatrix[o][d] <= MP_edge_limit){
                        
                        // edge index
                        MP_edge_index[0].push_back(o);
                        MP_edge_index[1].push_back(d);

                        // edge weights
                        MP_edge_weights.push_back(1 - (double)NodeCostMatrix[o][d]/(double)MP_edge_limit);

                        //edge locations
                        std::vector<edgeFeatures::PathNode> path = edgeFeatures::astar(env->nodes->locations[o], env->nodes->locations[d], env, &DefaultPlanner::global_neighbors);
                        for (int i=0; i<path.size(); i++){
                            int loc = path[i].loc;
                            edgeFeatures::Direction dir = path[i].dir;
                            MP_loc_to_edges_new[loc].push_back({MP_edge_index[0].size()-1, dir});
                        }
                        MP_edge_lengths_new.push_back(path.size());
                    }
                }
            }
        }
        for (int i=0; i<MP_edge_lengths_new.size(); i++){
            MP_edge_map[{MP_edge_index[0][i], MP_edge_index[1][i]}] = i;
        }
        std::cout << "Time for message passing and AStar: " << (double)(clock()-start)/CLOCKS_PER_SEC << std::endl;


        // node positions
        node_positions.resize(6, std::vector<double>(env->nodes->nNodes,0));
        for (int node=0; node<env->nodes->nNodes; node++){
            int loc = env->nodes->locations[node];
            int x = loc % env->cols;
            int y = loc / env->cols;
            double x_norm = (double)x/(double)(env->cols-1);
            double y_norm = (double)y/(double)(env->rows-1);
            node_positions[0][node] = x_norm;
            node_positions[1][node] = y_norm;
            node_positions[2][node] = (sin(x_norm*2*M_PI) + 1)/2;
            node_positions[3][node] = (cos(x_norm*2*M_PI) + 1)/2;
            node_positions[4][node] = (sin(y_norm*2*M_PI) + 1)/2;
            node_positions[5][node] = (cos(y_norm*2*M_PI) + 1)/2;
        }
    }

    ///////////////////////////////////////////////
    ///////////// Roadmap stuff ///////////////////

    std::vector<std::vector<int>> RM_AdjacencyMatrix;
    std::vector<std::vector<double>> RM_node_positions;

    if (observationTypes.count("roadmap-activation")){
        if (env->roadmap == nullptr){
            throw std::runtime_error("roadmap-activation observation requires roadmap to be loaded");
        }

        // roadmap adjacency matrix
        RM_AdjacencyMatrix = env->roadmap->graph.AdjacencyMatrix;

        // roadmap node positions
        RM_node_positions.resize(6, std::vector<double>(env->roadmap->graph.nNodes,0));
        for (int node=0; node<env->roadmap->graph.nNodes; node++){
            int loc = env->roadmap->graph.node_to_locs[node][0];
            int x = loc % env->cols;
            int y = loc / env->cols;
            double x_norm = (double)x/(double)(env->cols-1);
            double y_norm = (double)y/(double)(env->rows-1);
            RM_node_positions[0][node] = x_norm;
            RM_node_positions[1][node] = y_norm;
            RM_node_positions[2][node] = (sin(x_norm*2*M_PI) + 1)/2;
            RM_node_positions[3][node] = (cos(x_norm*2*M_PI) + 1)/2;
            RM_node_positions[4][node] = (sin(y_norm*2*M_PI) + 1)/2;
            RM_node_positions[5][node] = (cos(y_norm*2*M_PI) + 1)/2;
        }

    }
    
    return {max_num_agents, max_num_tasks, AdjacencyMatrix, NodeCostMatrix, MP_edge_index, MP_edge_weights, node_positions, MP_loc_to_edges_new, MP_edge_lengths_new, space_per_node_new, RM_AdjacencyMatrix, RM_node_positions};
}


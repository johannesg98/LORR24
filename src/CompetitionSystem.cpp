#include <cmath>
#include "CompetitionSystem.h"
#include <boost/tokenizer.hpp>
#include "SharedEnv.h"
#include "nlohmann/json.hpp"
#include <functional>
#include <Logger.h>
#include <heuristics.h>


using json = nlohmann::ordered_json;






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

    task_manager.update_tasks(curr_states, proposed_schedule, simulator.get_curr_timestep());

    logger->log_info("Step done.", simulator.get_curr_timestep());

    sync_shared_env();

    if (simulator.get_curr_timestep() == simulation_time){
        done = true;
    }

    return done;
}

pybind11::dict BaseSystem::get_reward(){
    pybind11::dict reward_dict;

    reward_dict["task-finished"] = task_manager.num_of_task_finish - num_of_task_finish_last_call;
    num_of_task_finish_last_call = task_manager.num_of_task_finish;
  
    reward_dict["A*-distance"] = env->Astar_reward;

    reward_dict["idle-agents"] = env->idle_agents_reward;

    reward_dict["tasks-assigned"] = env->tasks_assigned_reward;

    reward_dict["dist-reward"] = env->dist_reward;

    reward_dict["backtrack-rewards-first-errand"] = env->backtrack_rewards_first_errand;

    reward_dict["backtrack-rewards-whole-task"] = env->backtrack_rewards_whole_task;    
    
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



pybind11::dict BaseSystem::get_observation(std::unordered_set<std::string>& observationTypes){
    pybind11::dict obs;

    if (observationTypes.count("node-basics")){
        
        clock_t start = clock();

        // agents per node
        // free agents per node
        std::vector<int> agents_per_node(env->nodes->nNodes, 0);
        std::vector<int> free_agents_per_node(env->nodes->nNodes, 0);
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


        // distance until agent becomes available at node
        std::vector<int> distance_until_agent_available(env->nodes->nNodes, distance_until_agent_avail_MAX);
        for (int agent=0; agent<env->num_of_agents; agent++){
            int agent_loc = env->curr_states[agent].location;
            int task_id = env->curr_task_schedule[agent];
            if (task_id == -1){
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


        // time
        obs["time"] = env->curr_timestep;

        // congestion ratio at nodes
        std::vector<float> congestion_ratio_per_node(env->nodes->nNodes, 0);
        for (int i=0; i<env->nodes->nNodes; i++){
            if (space_per_node[i] > 0){
                congestion_ratio_per_node[i] = (float)agents_per_node[i]/(float)space_per_node[i];
            }
        }
        obs["congestion_ratio_per_node"] = congestion_ratio_per_node;
        
        

        // congestion at edges
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


        // agents tell closest task to nodes
        std::vector<int> contains_closest_task_per_node(env->nodes->nNodes, 0);
        std::vector<int> closest_task_connection_per_MP_edge(MP_edge_lengths.size(), 0);
        for (int agent=0; agent<env->num_of_agents; agent++){
            if (env->curr_task_schedule[agent] == -1){
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



        std::cout << "Time for getting observations: " << (double)(clock()-start)/CLOCKS_PER_SEC << std::endl;
    }

    return obs;
}

std::tuple<int,
            int,
            std::vector<std::vector<int>>,
            std::vector<std::vector<int>>,
            std::vector<std::vector<int>>,
            std::vector<double>,
            std::vector<std::vector<double>>,
            std::vector<std::vector<std::pair<int, edgeFeatures::Direction>>>,
            std::vector<int>,
            std::vector<int>
                                                                                    > BaseSystem::get_env_vals(int MP_edge_limit){

    int max_num_agents = env->num_of_agents;
    int max_num_tasks = env->task_pool.size();
    std::vector<std::vector<int>> AdjacencyMatrix = env->nodes->AdjacencyMatrix;


    // rebalancing optimizer cost matrix
    std::vector<std::vector<int>> NodeCostMatrix;
    NodeCostMatrix.resize(env->nodes->nNodes,std::vector<int>(env->nodes->nNodes,0));
    for (int i=0; i<env->nodes->nNodes; i++){
        int start_loc = env->nodes->nodes[i];
        for (int j=0; j<env->nodes->nNodes; j++){
            int target_loc = env->nodes->nodes[j];
            NodeCostMatrix[i][j] = DefaultPlanner::get_h(env, start_loc, target_loc);
        }
    }

    // count free spaces per node to calc congestion ratio
    std::vector<int> space_per_node_new(env->nodes->nNodes, 0);
    for (int id=0; id<env->map.size(); id++){
        int node = env->nodes->regions[id];
        if (env->map[id] == 0){
            space_per_node_new[node]++;
        }
    }


    // message passing
    clock_t start = clock();
    std::vector<std::vector<int>> MP_edge_index(2);
    std::vector<double> MP_edge_weights;
    std::vector<std::vector<std::pair<int, edgeFeatures::Direction>>> MP_loc_to_edges_new(env->map.size());
    std::vector<int> MP_edge_lengths_new;
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
                    std::vector<edgeFeatures::PathNode> path = edgeFeatures::astar(env->nodes->nodes[o], env->nodes->nodes[d], env, &DefaultPlanner::global_neighbors);
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
    std::vector<std::vector<double>> node_positions(6, std::vector<double>(env->nodes->nNodes,0));
    for (int node=0; node<env->nodes->nNodes; node++){
        int loc = env->nodes->nodes[node];
        int x = loc % env->cols;
        int y = loc / env->cols;
        double x_norm = (double)x/(double)env->cols;
        double y_norm = (double)y/(double)env->rows;
        node_positions[0][node] = x_norm;
        node_positions[1][node] = y_norm;
        node_positions[2][node] = (sin(x_norm*2*M_PI) + 1)/2;
        node_positions[3][node] = (cos(x_norm*2*M_PI) + 1)/2;
        node_positions[4][node] = (sin(y_norm*2*M_PI) + 1)/2;
        node_positions[5][node] = (cos(y_norm*2*M_PI) + 1)/2;
    }
    
    return {max_num_agents, max_num_tasks, AdjacencyMatrix, NodeCostMatrix, MP_edge_index, MP_edge_weights, node_positions, MP_loc_to_edges_new, MP_edge_lengths_new, space_per_node_new};
}


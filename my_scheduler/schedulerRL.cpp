#include "schedulerRL.h"

using namespace operations_research;

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
    clock_t start = clock();
    // cout<<"schedule plan limit" << time_limit <<endl;

    // the default scheduler keep track of all the free agents and unassigned (=free) tasks across timesteps
    free_agents.insert(env->new_freeagents.begin(), env->new_freeagents.end());
    free_tasks.insert(env->new_tasks.begin(), env->new_tasks.end());

    // std::cout << "SchedulerRL start, proposed_schedule: ";
    // for (int i = 0; i < proposed_schedule.size(); i++){
    //     std::cout << proposed_schedule[i] << " ";
    // }
    // std::cout << std::endl;

    if (free_agents.size() == 0){
        std::cout << "SchedulerRL end, no free agents" << std::endl;
        return;
    }

    int min_task_i, min_task_makespan, dist, agent_loc, task_loc, count, node_id;
    count = 0;

    std::map<pair<int,int>, int> reb_action = action_dict.at("reb_action").cast<std::map<pair<int,int>, int>>();

    std::vector<std::vector<int>> tasks_per_node(env->nodes->nNodes);
    for (int task_id : free_tasks){
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

    //copy tasks_per_node and fill with -1


    // create reservation list for each node for each task that indicates which task is reserved for which incoming node
    std::vector<std::vector<int>> task_reserved_for_incoming_node(tasks_per_node.size());
    for (size_t i = 0; i < tasks_per_node.size(); ++i) {
        task_reserved_for_incoming_node[i] = std::vector<int>(tasks_per_node[i].size(), -1);
    }

    // create a list of incoming agents for each node that contains tuple of agent id and origin node
    std::vector<std::vector<tuple<int, int>>> agent_a_incoming_from_node_b(env->nodes->nNodes);




    for (int node = 0; node < env->nodes->nNodes; node++){


        //check timeout
        // if (node % 10 == 0 && std::chrono::steady_clock::now() > endtime){
        //     std::cout << "schedulerRL terminated early due to timeout XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX part 1 node: " << node << std::endl;
        //     break;
        // }

        // Collect all incoming, staying and outgoing agent numbers and their origin and target nodes
        std::vector<int> outgoing_agents_targets;
        int staying_agents = 0;
        std::vector<int> incoming_agents_origins;
        for (int end = 0; end < env->nodes->nNodes; end++){
            if (reb_action[{node, end}] > 0){
                if (node == end){
                    staying_agents = reb_action[{node, end}];
                } else {
                    for (int i = 0; i < reb_action[{node, end}]; i++){
                        outgoing_agents_targets.push_back(end);
                    }
                }
            }
        }
        for (int start_node = 0; start_node < env->nodes->nNodes; start_node++){
            if (reb_action[{start_node, node}] > 0){
                if (start_node != node){
                    for (int i = 0; i < reb_action[{start_node, node}]; i++){
                        incoming_agents_origins.push_back(start_node);
                    }
                }
            }
        }

        // Problem size
        const int num_agents = outgoing_agents_targets.size() + staying_agents + incoming_agents_origins.size();
        const int num_tasks = tasks_per_node[node].size() + outgoing_agents_targets.size();

        assert(free_agents_per_node[node].size() == outgoing_agents_targets.size() + staying_agents);

        if (num_agents > 1){  // num_agents > 1, somehow the else part doesnt work. NumOfTaskFinished drops from 70 to 60. But doenst matter since its not much faster anway.
        
            // Create cost matrix
            std::vector<std::vector<int>> cost_matrix(num_agents, std::vector<int>(num_tasks, 0));

            // start with existing agents
            for (int agent = 0; agent < free_agents_per_node[node].size(); agent++){
                agent_loc = env->curr_states[free_agents_per_node[node][agent]].location;
                for (int task = 0; task < tasks_per_node[node].size(); task++){
                    task_loc = env->task_pool[tasks_per_node[node][task]].locations[0];
                    cost_matrix[agent][task] = DefaultPlanner::get_h(env, agent_loc, task_loc);
                }
                int already_filled_tasks = tasks_per_node[node].size();
                for (int task = 0; task < outgoing_agents_targets.size(); task++){
                    task_loc = env->nodes->nodes[outgoing_agents_targets[task]];
                    cost_matrix[agent][already_filled_tasks + task] = DefaultPlanner::get_h(env, agent_loc, task_loc);
                }
            }

            // add incoming agents
            int already_filled_agents = free_agents_per_node[node].size();
            for (int agent = 0; agent < incoming_agents_origins.size(); agent++){
                agent_loc = env->nodes->nodes[incoming_agents_origins[agent]];
                for (int task = 0; task < tasks_per_node[node].size(); task++){
                    task_loc = env->task_pool[tasks_per_node[node][task]].locations[0];
                    cost_matrix[already_filled_agents + agent][task] = DefaultPlanner::get_h(env, agent_loc, task_loc);
                }
                // incoming agents are not allowed to go to outgoing tasks, so no need to add these costs
            }

            // Create the solver
            std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("CBC"));
            if (!solver) {
                std::cerr << "CBC solver not available." << std::endl;
            }

            // Decsision variables
            std::vector<std::vector<const MPVariable*>> x(num_agents, std::vector<const MPVariable*>(num_tasks, nullptr));
            for (int i = 0; i < num_agents; ++i) {
                for (int j = 0; j < num_tasks; ++j) {
                    x[i][j] = solver->MakeIntVar(0, 1, "x_" + std::to_string(i) + "_" + std::to_string(j));
                                // MakeNumVar(0.0, 1.0, .....) for LP relaxation
                }
            }

            // Objective function: Minimize the total cost
            MPObjective* const objective = solver->MutableObjective();
            for (int i = 0; i < num_agents; ++i) {
                for (int j = 0; j < num_tasks; ++j) {
                    objective->SetCoefficient(x[i][j], cost_matrix[i][j]);
                }
            }
            objective->SetMinimization();

            // Constraint: number of all assignments = min(num_agents, num_tasks)
            int num_assignments = std::min(num_agents, num_tasks);
            LinearExpr total_assignments;
            for (int i = 0; i < num_agents; ++i) {
                for (int j = 0; j < num_tasks; ++j) {
                    total_assignments += x[i][j];
                }
            }
            solver->MakeRowConstraint(total_assignments == num_assignments);

            // Constraint: Dont allow incoming agents to take outgoing tasks
            LinearExpr banned_assignements;
            int already_filled_tasks = tasks_per_node[node].size();
            for (int agent = 0; agent < incoming_agents_origins.size(); agent++){
                for (int task = 0; task < outgoing_agents_targets.size(); task++){
                    banned_assignements += x[already_filled_agents + agent][already_filled_tasks + task];
                }
            }
            solver->MakeRowConstraint(banned_assignements == int(0));

            // Constraint: each agent is assigned to at most one task
            for (int i = 0; i < num_agents; ++i) {
                LinearExpr agent_constraint;
                for (int j = 0; j < num_tasks; ++j) {
                    agent_constraint += x[i][j];
                }
                solver->MakeRowConstraint(agent_constraint <= 1);
            }

            // Constraint: each task is assigned to at most one agent
            for (int j = 0; j < num_tasks; ++j) {
                LinearExpr task_constraint;
                for (int i = 0; i < num_agents; ++i) {
                    task_constraint += x[i][j];
                }
                solver->MakeRowConstraint(task_constraint <= 1);
            }

            // Solve
            const MPSolver::ResultStatus result_status = solver->Solve();
            
            // Check optimality
            if (result_status != MPSolver::OPTIMAL) {
                std::cerr << "The problem does not have an optimal solution!" << std::endl;
            }

            // Extract solution: start with existing (staying and outgoing) agents
            for (int agent = 0; agent < free_agents_per_node[node].size(); agent++){
                for (int task = 0; task < tasks_per_node[node].size(); task++){
                    if (x[agent][task]->solution_value() > 0.5){
                        proposed_schedule[free_agents_per_node[node][agent]] = tasks_per_node[node][task];
                        free_agents.erase(free_agents_per_node[node][agent]);
                        free_tasks.erase(tasks_per_node[node][task]);
                    }
                }
                int already_filled_tasks = tasks_per_node[node].size();
                for (int task = 0; task < outgoing_agents_targets.size(); task++){
                    if (x[agent][already_filled_tasks + task]->solution_value() > 0.5){
                        int target_node = outgoing_agents_targets[task];
                        int agent_id = free_agents_per_node[node][agent];
                        agent_a_incoming_from_node_b[target_node].push_back({agent_id, node});
                    }    
                }
            }

            // Extract solution: continue with incoming agents
            for (int agent = 0; agent < incoming_agents_origins.size(); agent++){
                for (int task = 0; task < tasks_per_node[node].size(); task++){
                    if (x[already_filled_agents + agent][task]->solution_value() > 0.5){
                        task_reserved_for_incoming_node[node][task] = incoming_agents_origins[agent];
                    }            
                }
                // incoming agents are not allowed to go to outgoing tasks, so no need to add these
            }
        }
        else if (num_agents == 1){
            // staying
            if (staying_agents == 1){
                int agent_id = free_agents_per_node[node][0];
                min_task_i = -1;
                min_task_makespan = INT_MAX;
                for (int task_id : tasks_per_node[node]){
                    task_loc = env->task_pool[task_id].locations[0];
                    dist = DefaultPlanner::get_h(env, env->curr_states[agent_id].location, task_loc);
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
            }
            //incoming
            else if (incoming_agents_origins.size() == 1){
                int start_node = incoming_agents_origins[0];
                int start_loc = env->nodes->nodes[start_node];
                min_task_i = -1;
                min_task_makespan = INT_MAX;
                for (int task = 0; task < tasks_per_node[node].size(); task++){
                    int task_id = tasks_per_node[node][task];
                    task_loc = env->task_pool[task_id].locations[0];
                    dist = DefaultPlanner::get_h(env, start_loc, task_loc);
                    if (dist < min_task_makespan){
                        min_task_i = task;
                        min_task_makespan = dist;
                    }
                }
                if (min_task_i != -1){
                    task_reserved_for_incoming_node[node][min_task_i] = start_node;
                }

            }
            //outgoing
            else{
                int agent_id = free_agents_per_node[node][0];
                int target_node = outgoing_agents_targets[0];
                agent_a_incoming_from_node_b[target_node].push_back({agent_id, node});
            }
                
            
        }    
        
    }


    for (int node = 0; node < env->nodes->nNodes; node++){

        //check timeout
        // if (node % 10 == 0 && std::chrono::steady_clock::now() > endtime){
        //     std::cout << "schedulerRL terminated early due to timeout XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX part 2 node: " << node << std::endl;
        //     break;
        // }
        for (auto [agent_id, origin_node] : agent_a_incoming_from_node_b[node]){
            int min_task = -1;
            min_task_makespan = INT_MAX;
            for (int task = 0; task < task_reserved_for_incoming_node[node].size(); task++){
                if (task_reserved_for_incoming_node[node][task] == origin_node){
                    dist = DefaultPlanner::get_h(env, env->curr_states[agent_id].location, env->task_pool[tasks_per_node[node][task]].locations[0]);
                    if (dist < min_task_makespan){
                        min_task = task;
                        min_task_makespan = dist;
                    }
                }
            }
            if (min_task != -1){
                proposed_schedule[agent_id] = tasks_per_node[node][min_task];
                free_agents.erase(agent_id);
                free_tasks.erase(tasks_per_node[node][min_task]);
                task_reserved_for_incoming_node[node][min_task] = -1;
            }
        }
    }

    

    // std::cout << "SchedulerRL end,   proposed_schedule: ";
    // for (int i = 0; i < proposed_schedule.size(); i++){
    //     std::cout << proposed_schedule[i] << " ";
    // }
    // std::cout << std::endl;




    // Shitty concept and there are also still mistakes in the code
    // Also doesnt account for the possibility of more agents in the start nodes than rebalancing agents towards the current node
    // // create a list of tasks per node
    // std::vector<std::vector<int>> tasks_per_node(env->nodes->nNodes);
    // for (int task_id : free_tasks){
    //     task_loc = env->task_pool[task_id].locations[0];
    //     node_id = env->nodes->regions[task_loc];
    //     tasks_per_node[node_id].push_back(task_id);
    // }
    // std::vector<std::vector<int>> free_agents_per_node(env->nodes->nNodes);
    // for (int agent_id : free_agents){
    //     agent_loc = env->curr_states[agent_id].location;
    //     node_id = env->nodes->regions[agent_loc];
    //     free_agents_per_node[node_id].push_back(agent_id);
    // }
    // bool timeout = false;
    // for (int goal_node = 0; goal_node < env->nodes->nNodes && !timeout; goal_node++){
    //     std::vector<int> start_nodes;
    //     for (int start_node = 0; start_node < env->nodes->nNodes; start_node++){
    //         if (start_node != goal_node){
    //             if(reb_action[{start_node, goal_node}] == 1){
    //                 start_nodes.push_back(start_node);
    //             }
    //         }
    //     }
    //     if (start_nodes.size() > 0 && tasks_per_node[goal_node].size() > 0){
    //         for (int agent_id : start_nodes){
    //             min_task_i = -1;
    //             min_task_makespan = INT_MAX;
    //             agent_loc = env->curr_states[agent_id].location;
    //             for (int task_id : tasks_per_node[goal_node]){
    //                 task_loc = env->task_pool[task_id].locations[0];
    //                 dist = DefaultPlanner::get_h(env, agent_loc, task_loc);
    //                 if (dist < min_task_makespan){
    //                     min_task_i = task_id;
    //                     min_task_makespan = dist;
    //                 }
    //             }
    //             if (min_task_i != -1){
    //                 proposed_schedule[agent_id] = min_task_i;
    //                 free_agents.erase(agent_id);
    //                 free_tasks.erase(min_task_i);
    //             }
    //             //check for timeout every 10 task evaluations
    //             if (count % 10 == 0 && std::chrono::steady_clock::now() > endtime){
    //                 timeout = true;
    //                 break;
    //             }
    //             count++;   
    //         }
    //     }
    // }
    cout << "Scheduler Time Usage: " <<  ((float)(clock() - start))/CLOCKS_PER_SEC <<endl;
    #ifndef NDEBUG    
    cout << "new free agents: " << env->new_freeagents.size() << " new tasks: "<< env->new_tasks.size() <<  endl;
    cout << "free agents: " << free_agents.size() << " free tasks: " << free_tasks.size() << endl;
    #endif
    return;
}
}

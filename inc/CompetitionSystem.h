#pragma once
// #include "BasicSystem.h"
#include "SharedEnv.h"
#include "Grid.h"
#include "Tasks.h"
#include "ActionModel.h"
#include "Entry.h"
#include "Logger.h"
#include "TaskManager.h"
#include <pthread.h>
#include <future>
#include "Simulator.h"

//RL added stuff
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "edge_features.h"
#include "Roadmap.h"

//NoManSky Solution
#include "schedulerNoMan.hpp"


struct pair_hash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        // Simple hash combine
        return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
    }
};



class BaseSystem
{
public:
    Logger* logger = nullptr;

	BaseSystem(Grid &grid, Entry* planner, std::vector<int>& start_locs, std::vector<list<int>>& tasks, ActionModelWithRotate* model):
      map(grid), planner(planner), env(planner->env),
      task_manager(tasks, start_locs.size()), simulator(grid,start_locs,model)
    {
        num_of_agents = start_locs.size();
        starts.resize(num_of_agents);

        for (size_t i = 0; i < start_locs.size(); i++)
            {
                if (grid.map[start_locs[i]] == 1)
                    {
                        cout<<"error: agent "<<i<<"'s start location is an obstacle("<<start_locs[i]<<")"<<endl;
                        exit(0);
                    }
                starts[i] = State(start_locs[i], 0, 0);
            }

 //        int task_id = 0;
 // for (auto& task_location: tasks)
 //        {
 //            all_tasks.emplace_back(task_id++, task_location);
 //            task_queue.emplace_back(all_tasks.back().task_id, all_tasks.back().locations.front());
 //            //task_queue.emplace_back(task_id++, task_location);
 //        }
 //        num_of_agents = start_locs.size();
 //        starts.resize(num_of_agents);
 //        for (size_t i = 0; i < start_locs.size(); i++)
 //        {
 //            starts[i] = State(start_locs[i], 0, 0);
 //        }
    };

	virtual ~BaseSystem()
    {
        //safely exit: wait for join the thread then delete planner and exit
        if (started)
        {
            task_td.join();
        }
        if (planner != nullptr)
        {
            delete planner;
        }
    };

    void set_num_tasks_reveal(float num){task_manager.set_num_tasks_reveal(num);};
    void set_plan_time_limit(int limit){plan_time_limit = limit;};
    void set_preprocess_time_limit(int limit){preprocess_time_limit = limit;};
    void set_log_level(int level){log_level = level;};
    void set_logger(Logger* logger){
        this->logger = logger;
        task_manager.set_logger(logger);
    }

    void simulate(int simulation_time);
    void plan(int & timeout_timesteps);
    bool planner_wrapper();

    //void saveSimulationIssues(const string &fileName) const;
    void saveResults(const string &fileName, int screen) const;

    //new functions for RL
    void initializeExtendedBaseSystem(int simulation_time);
    pybind11::dict get_NoManSkySolution(int time_limit = 100);
    bool step(const std::unordered_map<std::string, pybind11::object>& action_dict = {});
    void get_roadmap_reward(const std::vector<State>& curr_states);
    pybind11::dict get_reward();
    pybind11::dict get_info();
    int loadNodes(const std::string& fname);
    int loadRoadmapNodes(const std::string& fname);
    pybind11::dict get_observation(std::unordered_set<std::string>& observationTypes);
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
                                                                    > get_env_vals(std::unordered_set<std::string>& observationTypes, int MP_edge_limit = 0);
    int distance_until_agent_avail_MAX = 20;
    std::vector<std::vector<std::pair<int,edgeFeatures::Direction>>> MP_loc_to_edges;     // num_map_tiles x num_of_edges_that_pass_through_it x (edge_id, direction)
    std::vector<int> MP_edge_lengths;
    std::vector<int> space_per_node;
    
    MyScheduler schedulerNoMan;
    Roadmap roadmap;


protected:
    Grid map;
    int simulation_time;

    vector<Action> proposed_actions;
    vector<int> proposed_schedule;

    int total_timetous = 0;


    std::future<bool> future;
    std::thread task_td;
    bool started = false;

    Entry* planner;
    SharedEnvironment* env;

    int preprocess_time_limit=10;

    int plan_time_limit = 3;


    vector<State> starts;
    int num_of_agents;

    int log_level = 1;

    // tasks that haven't been finished but have been revealed to agents;

    vector<list<std::tuple<int,int,std::string>>> events;

    //for evaluation
    vector<int> solution_costs;
    list<double> planner_times; 
    bool fast_mover_feasible = true;


    void initialize();
    bool planner_initialize();


    TaskManager task_manager;
    Simulator simulator;
    // deque<Task> task_queue;
    virtual void sync_shared_env();

    void move(vector<Action>& actions);
    bool valid_moves(vector<State>& prev, vector<Action>& next);

    void log_preprocessing(bool succ);
    // void log_event_assigned(int agent_id, int task_id, int timestep);
    // void log_event_finished(int agent_id, int task_id, int timestep);

    //new functions for RL
    int num_of_task_finish_last_call = 0;
    int num_of_first_errands_started_last_call = 0;
    int length_of_tasks_finished_last_call = 0;
    std::unordered_map<std::pair<int, int>, int, pair_hash> MP_edge_map;
    vector<State> last_agent_states;

    // roadmap
    int last_roadmap_distance_sum;
    std::vector<int> last_agent_goals;
    int roadmap_progress_reward;
    
    

};


// class TaskAssignSystem : public BaseSystem
// {
// public:
// 	TaskAssignSystem(Grid &grid, MAPFPlanner* planner, std::vector<int>& start_locs, std::vector<int>& tasks, ActionModelWithRotate* model):
//         BaseSystem(grid, planner, model)
//     {
//         int task_id = 0;
//         for (auto& task_location: tasks)
//         {
//             all_tasks.emplace_back(task_id++, task_location);
//             task_queue.emplace_back(all_tasks.back().task_id, all_tasks.back().locations.front());
//             //task_queue.emplace_back(task_id++, task_location);
//         }
//         num_of_agents = start_locs.size();
//         starts.resize(num_of_agents);
//         for (size_t i = 0; i < start_locs.size(); i++)
//         {
//             starts[i] = State(start_locs[i], 0, 0);
//         }
//     };

// 	~TaskAssignSystem(){};


// private:
//     deque<Task> task_queue;

// 	void update_tasks();
// };



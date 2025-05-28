#pragma once
#include "Tasks.h"
#include "SharedEnv.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "schedulerNoMan.hpp"
#include <chrono>


class TaskScheduler
{
    public:
        SharedEnvironment* env;
        std::vector<int> task_search_start_times;
        std::string backtrack_reward_type = "MaxDist-Time";

        TaskScheduler(SharedEnvironment* env): env(env){};
        TaskScheduler(){env = new SharedEnvironment();};
        virtual ~TaskScheduler(){delete env;};
        virtual void initialize(int preprocess_time_limit);
        virtual void plan(int time_limit, std::vector<int> & proposed_schedule, const std::unordered_map<std::string, pybind11::object>& action_dict = {});

        MyScheduler schedulerNoMan;

        int solveTimeSum = 0;
        int solveCount = 0;
        std::string scheduler_type;
        
};
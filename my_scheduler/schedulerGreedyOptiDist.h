#ifndef GreedyOptiDist
#define GreedyOptiDist

#include "Types.h"
#include "SharedEnv.h"
#include "heuristics.h"
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace schedulerGreedyOptiDist{

void schedule_initialize(int preprocess_time_limit, SharedEnvironment* env);

void schedule_plan(int time_limit, std::vector<int> & proposed_schedule,  SharedEnvironment* env, const std::unordered_map<std::string, pybind11::object>& action_dict);

}

#endif
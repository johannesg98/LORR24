#ifndef SCHEDULERRL
#define SCHEDULERRL

#include "Types.h"
#include "SharedEnv.h"
#include "heuristics.h"
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "planner.h"
#include "search.h"

// namespace DefaultPlanner{
//     extern std::vector<int> decision; 
//     extern std::vector<int> prev_decision;
//     extern std::vector<double> p;
//     extern std::vector<State> prev_states;
//     extern std::vector<State> next_states;
//     extern std::vector<int> ids;
//     extern std::vector<double> p_copy;
//     extern std::vector<bool> occupied;
//     extern std::vector<DCR> decided;
//     extern std::vector<bool> checked;
//     extern std::vector<bool> require_guide_path;
//     extern std::vector<int> dummy_goals;
//     extern TrajLNS trajLNS;
//     extern std::vector<double> t;
// }

namespace schedulerRL{

void reset_globals();

void schedule_initialize(int preprocess_time_limit, SharedEnvironment* env);

void schedule_plan(int time_limit, std::vector<int> & proposed_schedule,  SharedEnvironment* env, const std::unordered_map<std::string, pybind11::object>& action_dict = {});

}

#endif
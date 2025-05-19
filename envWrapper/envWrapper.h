#ifndef LRR_ENV_H
#define LRR_ENV_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <memory>
#include <string>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include "../inc/nlohmann/json.hpp"
#include "../inc/CompetitionSystem.h"
#include "../inc/Evaluation.h"
#include "../my_scheduler/schedulerRL.h"

namespace po = boost::program_options;
using json = nlohmann::json;

class LRRenv {
private:
    bool done;
    int step_count;
    Logger* logger = nullptr;
    ActionModelWithRotate* model = nullptr;
    std::optional<Grid> grid;
    std::vector<int> agents;
    std::vector<list<int>> tasks;
    std::string rewardType;
    std::unordered_set<std::string> observationTypes;
    std::string random_agents_and_tasks;
    int message_passing_edge_limit;
    int distance_until_agent_avail_MAX;
    json data;
    bool is_initialized = false;
    std::vector<std::vector<std::pair<int,edgeFeatures::Direction>>> MP_loc_to_edges;     // num_map_tiles x num_of_edges_that_pass_through_it x (edge_id, direction)
    std::vector<int> MP_edge_lengths;

    // Command-line arguments stored as class variables
    std::string inputFile;
    std::string outputFile;
    int outputScreen;
    bool evaluationMode;
    int simulationTime;
    std::string fileStoragePath;
    int planTimeLimit;
    int preprocessTimeLimit;
    std::string logFile;
    int logDetailLevel;

    // Variables as in driver.cpp 
    std::unique_ptr<BaseSystem> system_ptr;

public:
    // Constructor with parameters
    LRRenv(
        std::string inputFile,
        std::string outputFile = "./outputs/continousTrainOutput.json",
        int outputScreen = 1,
        bool evaluationMode = false,
        int simulationTime = 600,
        std::string fileStoragePath = "",
        int planTimeLimit = 10000,
        int preprocessTimeLimit = 30000,
        std::string logFile = "",
        int logDetailLevel = 1,
        std::string rewardType = "task-finished",
        std::unordered_set<std::string> observationTypes = {},
        std::string random_agents_and_tasks = "true",
        int message_passing_edge_limit = 0,
        int distance_until_agent_avail_MAX = 20
    );

    // Function declarations
    std::tuple<pybind11::dict, double, bool> reset(
        std::string inputFile_ = "",
        std::string outputFile_ = "",
        int outputScreen_ = -1,
        bool evaluationMode_ = false,
        int simulationTime_ = -1,
        std::string fileStoragePath_ = "",
        int planTimeLimit_ = -1,
        int preprocessTimeLimit_ = -1,
        std::string logFile_ = "",
        int logDetailLevel_ = -1,
        std::string rewardType_ = "invalid",
        std::unordered_set<std::string> observationTypes_ = {"-1"},
        std::string random_agents_and_tasks_ = "no_input",
        int message_passing_edge_limit_ = 0,
        int distance_until_agent_avail_MAX_ = -1
    );
    std::tuple<pybind11::dict, pybind11::dict, bool, pybind11::dict> step(const std::unordered_map<std::string, pybind11::object>& action_dict = {});
    void make_env_params_available();
    int nNodes = -1;
    int nAgents = -1;
    int nTasks = -1;
    std::vector<std::vector<int>> AdjacencyMatrix;
    std::vector<std::vector<int>> NodeCostMatrix;
    std::vector<std::vector<int>> MP_edge_index;
    std::vector<double> MP_edge_weights;
    std::vector<std::vector<double>> node_positions;
    std::vector<int> space_per_node;
};

#endif // LRR_ENV_H

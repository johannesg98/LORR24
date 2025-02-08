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
#include "ExtendedBaseSystem.h"
#include "../inc/Evaluation.h"

namespace po = boost::program_options;
using json = nlohmann::json;

class LRRenv {
private:
    bool done;
    int step_count;
    std::vector<double> state;
    Logger* logger = nullptr;
    ActionModelWithRotate* model = nullptr;
    std::optional<Grid> grid;
    std::vector<int> agents;
    std::vector<list<int>> tasks ;

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
    std::unique_ptr<ExtendedBaseSystem> system_ptr;

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
        int logDetailLevel = 1
    );

    // Function declarations
    void reset(
        std::string inputFile_ = "",
        std::string outputFile_ = "",
        int outputScreen_ = -1,
        bool evaluationMode_ = false,
        int simulationTime_ = -1,
        std::string fileStoragePath_ = "",
        int planTimeLimit_ = -1,
        int preprocessTimeLimit_ = -1,
        std::string logFile_ = "",
        int logDetailLevel_ = -1
    );
    std::tuple<std::vector<double>, double, bool> step();
};

#endif // LRR_ENV_H

#include "envWrapper.h"
#include <iostream>
#include <algorithm>
#include <random>


// Function to read the first line in a file
int read_first_line_as_int(const std::string& filename) {
    std::ifstream file(filename);
    int first_line_value;
    file >> first_line_value;
    return first_line_value;
}

// constructor
LRRenv::LRRenv(
                std::string inputFile,
                std::string outputFile,
                int outputScreen,
                bool evaluationMode,
                int simulationTime,
                std::string fileStoragePath,
                int planTimeLimit,
                int preprocessTimeLimit,
                std::string logFile,
                int logDetailLevel,
                std::string rewardType,
                std::unordered_set<std::string> observationTypes,
                std::string random_agents_and_tasks,
                int message_passing_edge_limit,
                int distance_until_agent_avail_MAX
                ) 
                : done(false), step_count(0), inputFile(inputFile), outputFile(outputFile), outputScreen(outputScreen), evaluationMode(evaluationMode),
                simulationTime(simulationTime), fileStoragePath(fileStoragePath), planTimeLimit(planTimeLimit), preprocessTimeLimit(preprocessTimeLimit),
                logFile(logFile), logDetailLevel(logDetailLevel), rewardType(rewardType), observationTypes(std::move(observationTypes)),
                random_agents_and_tasks(random_agents_and_tasks), message_passing_edge_limit(message_passing_edge_limit), distance_until_agent_avail_MAX(distance_until_agent_avail_MAX)
{
    std::cout << "Environment constructed" << std::endl;

    auto input_json_file = inputFile;
    std::ifstream f(input_json_file);
    try{
        data = json::parse(f);
    }
    catch (json::parse_error error){
        std::cerr << "Failed to load " << input_json_file << std::endl;
        std::cerr << "Message: " << error.what() << std::endl;
        exit(1);
    }
}

// reset function with optional arguments to change environment
std::tuple<pybind11::dict, double, bool> LRRenv::reset(
                                                        std::string inputFile_, std::string outputFile_, int outputScreen_,
                                                        bool evaluationMode_, int simulationTime_, std::string fileStoragePath_,
                                                        int planTimeLimit_, int preprocessTimeLimit_, std::string logFile_, int logDetailLevel_, std::string rewardType_,
                                                        std::unordered_set<std::string> observationTypes_, std::string random_agents_and_tasks_, int message_passing_edge_limit_,
                                                        int distance_until_agent_avail_MAX_
                                                        )
{   
    std::cout << "reset started cpp" << std::endl;

    done = false;
    step_count = 0;

    // overwrite existing environment arguments with optional new ones
    if (!inputFile_.empty()){
        inputFile = inputFile_;
        auto input_json_file = inputFile;
        std::ifstream f(input_json_file);
        try{
            data = json::parse(f);
        }
        catch (json::parse_error error){
            std::cerr << "Failed to load " << input_json_file << std::endl;
            std::cerr << "Message: " << error.what() << std::endl;
            exit(1);
        }
    }
    if (!outputFile_.empty()) outputFile = outputFile_;
    if (outputScreen_ != -1) outputScreen = outputScreen_;
    if (evaluationMode_) evaluationMode = evaluationMode_;
    if (simulationTime_ != -1) simulationTime = simulationTime_;
    if (!fileStoragePath_.empty()) fileStoragePath = fileStoragePath_;
    if (planTimeLimit_ != -1) planTimeLimit = planTimeLimit_;
    if (preprocessTimeLimit_ != -1) preprocessTimeLimit = preprocessTimeLimit_;
    if (!logFile_.empty()) logFile = logFile_;
    if (logDetailLevel_ != -1) logDetailLevel = logDetailLevel_;
    if (rewardType_ != "invalid") rewardType = rewardType_;
    if (!observationTypes_.count("-1")) observationTypes = observationTypes_;
    if (random_agents_and_tasks_ != "no_input") random_agents_and_tasks = random_agents_and_tasks_;
    if (message_passing_edge_limit_ != 0) message_passing_edge_limit = message_passing_edge_limit_;
    if (distance_until_agent_avail_MAX_ != -1) distance_until_agent_avail_MAX = distance_until_agent_avail_MAX_;

    // create base folder as in driver.cpp
    boost::filesystem::path p(inputFile);
    boost::filesystem::path dir = p.parent_path();
    std::string base_folder = dir.string();
    if (base_folder.size() > 0 && base_folder.back() != '/'){
        base_folder += "/";
    }

    // craete logger as in driver.cpp 
    int log_level = logDetailLevel;
    if (log_level <= 1)
        log_level = 2; //info
    else if (log_level == 2)
        log_level = 3; //warning
    else
        log_level = 5; //fatal
    logger = new Logger(logFile,log_level);
    std::filesystem::path filepath(outputFile);
    if (filepath.parent_path().string().size() > 0 && !std::filesystem::is_directory(filepath.parent_path())){
        logger->log_fatal("output directory does not exist",0);
        _exit(1);
    }

    // create Entry instance as in driver.cpp
    Entry *planner = new Entry();

    // load map as in driver.cpp
    auto map_path = read_param_json<std::string>(data, "mapFile");
    // Grid grid(base_folder + map_path);
    grid.emplace(base_folder + map_path);
    
    planner->env->map_name = map_path.substr(map_path.find_last_of("/") + 1);

    // create large file storage as in driver.cpp
    string file_storage_path = fileStoragePath;
    if (file_storage_path==""){
      char const* tmp = getenv("LORR_LARGE_FILE_STORAGE_PATH");
      if ( tmp != nullptr ) {
        file_storage_path = string(tmp);
      }
    }
    if (file_storage_path!="" &&!std::filesystem::exists(file_storage_path)){
      std::ostringstream stringStream;
      stringStream << "fileStoragePath (" << file_storage_path << ") is not valid";
      logger->log_warning(stringStream.str());
    }
    planner->env->file_storage_path = file_storage_path;

    // create simulation system as in driver.cpp
    model = new ActionModelWithRotate(*grid);
    model->set_logger(logger);

    int team_size = read_param_json<int>(data, "teamSize");
    // agents = read_int_vec(base_folder + read_param_json<std::string>(data, "agentFile"), team_size);
    
    std::string agent_file = base_folder + read_param_json<std::string>(data, "agentFile");
    int total_agents = read_first_line_as_int(agent_file);
    std::vector<int> all_agents = read_int_vec(agent_file, total_agents);

    tasks = read_int_vec(base_folder + read_param_json<std::string>(data, "taskFile"));

    

    // Shuffle agents and tasks
    if (random_agents_and_tasks == "true"){
        std::cout << "Shuffling agents and tasks" << std::endl;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(all_agents.begin(), all_agents.end(), g);
        std::shuffle(tasks.begin(), tasks.end(), g);
    }

    agents.assign(all_agents.begin(), all_agents.begin() + team_size);

    if (agents.size() > tasks.size())
        logger->log_warning("Not enough tasks for robots (number of tasks < team size)");
    system_ptr = std::make_unique<BaseSystem>(*grid, planner, agents, tasks, model);

    //add parameters as in driver.cpp
    system_ptr->set_logger(logger);
    system_ptr->set_plan_time_limit(planTimeLimit);
    system_ptr->set_preprocess_time_limit(preprocessTimeLimit);
    system_ptr->set_num_tasks_reveal(read_param_json<float>(data, "numTasksReveal", 1));

    //new functions for RL
    if (observationTypes.count("node-basics")){
        nNodes = system_ptr->loadNodes(base_folder + read_param_json<std::string>(data, "nodeFile"));
        system_ptr->distance_until_agent_avail_MAX = distance_until_agent_avail_MAX;
    }

    //initializes the environment as in BaseSystem::simulate
    system_ptr->initializeExtendedBaseSystem(simulationTime);


    double reward = 0.0;
    pybind11::dict obs;
    if (is_initialized){
        system_ptr->MP_loc_to_edges = MP_loc_to_edges;
        system_ptr->MP_edge_lengths = MP_edge_lengths;
        system_ptr->space_per_node = space_per_node;
        
        obs = system_ptr->get_observation(observationTypes);
    }
    is_initialized = true;

    

    std::cout << "reset done cpp" << std::endl;
    return {obs, reward, done};
}


std::tuple<pybind11::dict, pybind11::dict, bool, pybind11::dict> LRRenv::step(const std::unordered_map<std::string, pybind11::object>& action_dict) {
    done = system_ptr->step(action_dict);


    pybind11::dict reward = system_ptr->get_reward();
    pybind11::dict info = system_ptr->get_info();
    pybind11::dict obs = system_ptr->get_observation(observationTypes);

    if (done){
        system_ptr->saveResults(outputFile,outputScreen);
        delete model;
        delete logger;
        std::cout << "Environment closed and results saved in: " << outputFile << std::endl;
    }

    return {obs, reward, done, info};
}


void LRRenv::make_env_params_available(){
    reset();
    std::tie(nAgents, nTasks, AdjacencyMatrix, NodeCostMatrix, MP_edge_index, MP_edge_weights, node_positions, MP_loc_to_edges, MP_edge_lengths, space_per_node) = system_ptr->get_env_vals(message_passing_edge_limit);
    return;
}


// Bindings to Python
PYBIND11_MODULE(envWrapper, m) {
    pybind11::class_<LRRenv>(m, "LRRenv")
        .def(pybind11::init<
            std::string, std::string, int, bool, int, std::string, int, int, std::string, int, std::string, std::unordered_set<std::string>, std::string, int, int>(),
            pybind11::arg("inputFile"),
            pybind11::arg("outputFile") = "./outputs/pyTest.json",
            pybind11::arg("outputScreen") = 1,
            pybind11::arg("evaluationMode") = false,
            pybind11::arg("simulationTime") = 100,
            pybind11::arg("fileStoragePath") = "",
            pybind11::arg("planTimeLimit") = 1000,
            pybind11::arg("preprocessTimeLimit") = 30000,
            pybind11::arg("logFile") = "",
            pybind11::arg("logDetailLevel") = 1,
            pybind11::arg("rewardType") = "task-finished",
            pybind11::arg("observationTypes") = std::unordered_set<std::string>(),
            pybind11::arg("random_agents_and_tasks") = "true",
            pybind11::arg("message_passing_edge_limit") = 0,
            pybind11::arg("distance_until_agent_avail_MAX") = 20
        )
        .def("reset", &LRRenv::reset,
            pybind11::arg("inputFile_") = "",
            pybind11::arg("outputFile_") = "",
            pybind11::arg("outputScreen_") = -1,
            pybind11::arg("evaluationMode_") = false,
            pybind11::arg("simulationTime_") = -1,
            pybind11::arg("fileStoragePath_") = "",
            pybind11::arg("planTimeLimit_") = -1,
            pybind11::arg("preprocessTimeLimit_") = -1,
            pybind11::arg("logFile_") = "",
            pybind11::arg("logDetailLevel_") = -1,
            pybind11::arg("rewardType_") = "invalid",
            pybind11::arg("observationTypes_") = std::unordered_set<std::string>{"-1"},
            pybind11::arg("random_agents_and_tasks_") = "no_input",
            pybind11::arg("message_passing_edge_limit_") = 0,
            pybind11::arg("distance_until_agent_avail_MAX_") = -1
        )
        .def("step", &LRRenv::step, 
            pybind11::arg("reb_action") = pybind11::dict())   
        .def("make_env_params_available", &LRRenv::make_env_params_available)
        .def_readwrite("nNodes", &LRRenv::nNodes)
        .def_readwrite("nAgents", &LRRenv::nAgents)
        .def_readwrite("nTasks", &LRRenv::nTasks)
        .def_readwrite("AdjacencyMatrix", &LRRenv::AdjacencyMatrix)
        .def_readwrite("NodeCostMatrix", &LRRenv::NodeCostMatrix)
        .def_readwrite("MP_edge_index", &LRRenv::MP_edge_index)
        .def_readwrite("MP_edge_weights", &LRRenv::MP_edge_weights)
        .def_readwrite("node_positions", &LRRenv::node_positions);
}

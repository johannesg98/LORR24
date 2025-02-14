#include "envWrapper.h"
#include <iostream>

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
    RewardType rewardType,
    std::unordered_set<std::string> observationTypes
) 
: done(false), step_count(0), 
  inputFile(inputFile), outputFile(outputFile), 
  outputScreen(outputScreen), evaluationMode(evaluationMode),
  simulationTime(simulationTime), fileStoragePath(fileStoragePath),
  planTimeLimit(planTimeLimit), preprocessTimeLimit(preprocessTimeLimit),
  logFile(logFile), logDetailLevel(logDetailLevel), rewardType(rewardType),
  observationTypes(std::move(observationTypes)) {
    std::cout << "Environment constructed" << std::endl;
}

// reset function with optional arguments to change environment
std::tuple<pybind11::dict, double, bool> LRRenv::reset(
    std::string inputFile_, std::string outputFile_, int outputScreen_,
    bool evaluationMode_, int simulationTime_, std::string fileStoragePath_,
    int planTimeLimit_, int preprocessTimeLimit_, std::string logFile_, int logDetailLevel_, RewardType rewardType_,
    std::unordered_set<std::string> observationTypes_
) {
    std::cout << "reset started cpp" << std::endl;

    done = false;
    step_count = 0;

    // overwrite existing environment arguments with optional new ones
    if (!inputFile_.empty()) inputFile = inputFile_;
    if (!outputFile_.empty()) outputFile = outputFile_;
    if (outputScreen_ != -1) outputScreen = outputScreen_;
    if (evaluationMode_) evaluationMode = evaluationMode_;
    if (simulationTime_ != -1) simulationTime = simulationTime_;
    if (!fileStoragePath_.empty()) fileStoragePath = fileStoragePath_;
    if (planTimeLimit_ != -1) planTimeLimit = planTimeLimit_;
    if (preprocessTimeLimit_ != -1) preprocessTimeLimit = preprocessTimeLimit_;
    if (!logFile_.empty()) logFile = logFile_;
    if (logDetailLevel_ != -1) logDetailLevel = logDetailLevel_;
    if (rewardType_ != RewardType::INVALID) rewardType = rewardType_;
    if (!observationTypes_.count("-1")) observationTypes = observationTypes_;

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
    auto input_json_file = inputFile;
    json data;
    std::ifstream f(input_json_file);
    try{
        data = json::parse(f);
    }
    catch (json::parse_error error){
        std::cerr << "Failed to load " << input_json_file << std::endl;
        std::cerr << "Message: " << error.what() << std::endl;
        exit(1);
    }
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
    agents = read_int_vec(base_folder + read_param_json<std::string>(data, "agentFile"), team_size);
    tasks = read_int_vec(base_folder + read_param_json<std::string>(data, "taskFile"));
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
        system_ptr->loadNodes(base_folder + read_param_json<std::string>(data, "nodeFile"));
    }

    //initializes the environment as in BaseSystem::simulate
    system_ptr->initializeExtendedBaseSystem(simulationTime);

    //get obs,reward,done

    

    double reward = 0.0;
    pybind11::dict obs = system_ptr->get_observation(observationTypes);

    std::cout << "reset done cpp" << std::endl;
    return {obs, reward, done};


    //for this function: just try to figure out how to return the obs, reward, done, etc. from the env that is set up
    //maybe do it in initialize if not accessable otherwise

    //in step: just put sync_shared_env(); and planner->compute()  and then the move stuff from simulate for loop.
    // most other things can probably be ignored. Especially keep "started" just always false (default), then it probably works

    //dunno if we need to close something in the end. Just look what simulate() and driver.cpp do afterwards

}

std::tuple<pybind11::dict, double, bool> LRRenv::step() {

    done = system_ptr->step();

    double reward = system_ptr->get_reward(RewardType::TASKFINISHED);
    pybind11::dict obs = system_ptr->get_observation(observationTypes);

    if (done){
        system_ptr->saveResults(outputFile,outputScreen);
        delete model;
        delete logger;
    }

    return {obs, reward, done};
}


// Bindings to Python
PYBIND11_MODULE(envWrapper, m) {
    pybind11::enum_<RewardType>(m, "RewardType")
        .value("TASKFINISHED", RewardType::TASKFINISHED)
        .value("INVALID", RewardType::INVALID)
        .export_values();
    pybind11::class_<LRRenv>(m, "LRRenv")
        .def(pybind11::init<
            std::string, std::string, int, bool, int, std::string, int, int, std::string, int, RewardType, std::unordered_set<std::string>>(),
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
            pybind11::arg("rewardType") = RewardType::TASKFINISHED,
            pybind11::arg("observationTypes") = std::unordered_set<std::string>()
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
            pybind11::arg("rewardType_") = RewardType::INVALID,
            pybind11::arg("observationTypes_") = std::unordered_set<std::string>{"-1"}
        )
        .def("step", &LRRenv::step);
}

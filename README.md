# Installation
## Dependencies
- [cmake >= 3.16](https://cmake.org/)
- [libboost >= 1.49.0](https://www.boost.org/)
- Python3 >= 3.11 and [pybind11](https://pybind11.readthedocs.io/en/stable/) >=2.10.1

### Ubuntu:
```shell
sudo apt-get update
sudo apt-get install build-essential libboost-all-dev python3-dev python3-pybind11 
```

### Windows
1. Install WSL
2. Install necessary tools (CMake, GCC, Boost, pip, Pybind11):
```shell
sudo apt-get update
sudo apt-get install cmake g++ libboost-all-dev python3-dev python3-pip
pip install pybind11-global numpy
```

## Install python requirements
```shell
python3 -m venv RL/venv
source RL/venv/bin/activate
pip install -r RL/requirements.txt
```


## Build environment wrapper
```shell
cd envWrapper
mkdir build
cd build
cmake ..
make -j8
cd ../..
```


# Overview
The folder structure is based on the github repository of the League of Robot Runners. Additional parts for learning are integrated.
The main folder (LORR24) contains the folders src, inc and default_planner which contain the source code provided by the LRR as well as the CMakeLists.txt for the LRR build.
The folder example_problems contains all maps and simulation scenarios including agents and tasks.
The gymnasium style python wrapper is located with source code and build folder in envWrapper. This folder also contains the test-script to execute different schedulers easily.
Everything related to Reinforcement Learning is found in RL.

# Training
For training the RL-controller, the config script can be found in RL/src/config/model/sac.yaml
The default setting includes 200 agents and can be trained right away with running:
```shell
python3 RL/train.py
```
We provide 4 different warehouse maps:
- warehouse_6x4 --> contains 100 agents, mapsize: 589
- warehouse_8x6 --> contains 200 agents, mapsize: 975
- warehouse_9x8 --> contains 300 agents, mapsize: 1333
- warehouse_13x12 --> contains 500 agents, mapsize: 2537

This can be set with choosing the "map_path" (line 10) in the sac.yaml config file.
The file name of the output weights can be set in the config file with "checkpoint_path" (line 71). The file will be saved in RL/ckpt.
Further changes of the map, agents, tasks, etc. can be made in example_problems/custom_warehouse.domain.

For tracking the training progress, we recommend WandB. It can be activated in the config file (line 65) and configured in RL/train.py (line 79-89).

# Testing
## RL
Testing the RL-controller can be done with:
```shell
python3 RL/test.py
```
In the same config file as for training, the map needs to be chosen with "map_path" (line 10).
For the 4 provided training settings, we provide final training weights in RL/example_checkpoints.
They need to be specified accordingly in the config file in "load_test_checkpoint_path" (line 42).
Other relevant options for testing are found there as well. To make results comparable with the LRR, a computation time per step of 1000ms should be choosen. But for quick testing, 70ms is enough. The task-scheduler and path-planner for these small maps mostly converge in that time anyway. The throughput should be pretty much the same.

## Other schedulers
To compare the results to the default greedy, ILP and LRR-winner (NoManSky), a testing script can be used with:
```shell
python3 envWrapper/testEnv.py
```
Interesting options to change in the script are:
- Line 13 - inputFile: Choice of warehouse map/setting
- Line 15 - simulationTime: 10000 simuation steps in lifelong setting
- Line 16 - planTimeLimit: 1000ms in LRR, 70ms for quick testing (results should be the same again)
- Line 20 - scheduler_type: Choice of task-scheduler
- Line 28 - number_of_runs: Select the number of test runs. For lifelong, 1 run is often significant enough.



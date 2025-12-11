# Installation
1. Clone this repo
## Dependencies
- [cmake >= 3.16](https://cmake.org/)
- [libboost >= 1.49.0](https://www.boost.org/)
- Python3 >= 3.11 (mainly tested 3.12.3) and [pybind11](https://pybind11.readthedocs.io/en/stable/) >=2.10.1

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
```

## Install python requirements
```shell
python3 -m venv GRAND/RL/venv
source GRAND/RL/venv/bin/activate
pip install -r GRAND/RL/requirements.txt
```


## Build environment wrapper
```shell
cd League_of_Robot_Runners/envWrapper
mkdir build
cd build
cmake ..
make -j8
cd ../../..
```


# Overview
The folders are structured the following:

GRAND contains everything related to our method, split in three steps as in the paper. These are
1. Macroscopic Guidance (Reinforcement Learning (RL))
2. Rebalancing
3. Microscopic Assignments (Matching)
The relevant RL part i located in the learn function in GRAND/RL/src/algos/sac.py from line 558 on.

League_of_Robot_Runners contains the simulation from the League of Robot Runners (LRR) as well as a gymnasium style python wrapper, adjusted for RL.

# Training
For training the RL-controller, the config script can be found in GRAND/RL/src/config/model/sac.yaml.
The default setting includes 200 agents and can be trained right away with running (preferably on GPU):
```shell
python3 GRAND_train.py
```
We provide 4 different warehouse maps:
- warehouse_6x4 --> contains 100 agents, mapsize: 589
- warehouse_8x6 --> contains 200 agents, mapsize: 975
- warehouse_9x8 --> contains 300 agents, mapsize: 1333
- warehouse_13x12 --> contains 500 agents, mapsize: 2537

This can be set with choosing the "map_path" (line 9) in the sac.yaml config file.
The file name of the output weights can be set in the config file with "checkpoint_path" (line 70). The file will be saved in GRAND/RL/ckpt.
Further changes of the map, agents, tasks, etc. can be made in League_of_Robot_Runners/example_problems/custom_warehouse.domain.
(6x4, 8x6, etc. stands for the number of storage shelf blocks)

For tracking the training progress, we recommend WandB. It can be activated in the config file (line 64) and configured in GRAND_train.py (line 70-75).

# Testing
## RL
Testing the RL-controller can be done right away (default 200 agents, 10,000 steps) with:
```shell
python3 GRAND_test.py
```
In the same config file as for training, the map can be chosen with "map_path" (line 9).
For the 4 provided training settings, we provide final training weights in GRAND/RL/example_checkpoints.
They need to be specified accordingly in the config file in "load_test_checkpoint_path" (line 41).
Other relevant options for testing are found there as well. To make results comparable with the LRR, a computation time per step of 1000ms should be choosen. But for quick testing, 70ms is enough. The task-scheduler and path-planner for these small maps mostly converge in that time anyway. The resulting throughput should be similar.

## Other schedulers
To compare the results to the default greedy, G-OPT (ILP) and LRR Winner (NoManSky), a testing script can be used with:
```shell
python3 schedulers_test.py
```
Relevant options to change in the script are:
- Line 13 - inputFile: Choice of warehouse map/setting
- Line 15 - simulationTime: 10000 simuation steps in lifelong setting
- Line 16 - planTimeLimit: 1000ms in LRR, 70ms for quick testing (results should be the same again)
- Line 20 - scheduler_type: Choice of task-scheduler
- Line 28 - number_of_runs: Select the number of test runs. For lifelong (10000 steps), 1 run is often significant enough.



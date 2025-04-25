import sys
import os
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(script_dir, "../../../envWrapper/build")
sys.path.append(build_path)

src_path = os.path.join(script_dir, "../..")
sys.path.append(src_path)

import envWrapper
from src.helperfunctions.LRRParser import LRRParser
from src.helperfunctions.skip_actor import skip_actor
from src.algos.reb_flow_solver import solveRebFlow
from src.helperfunctions.assign_discrete_actions import assign_discrete_actions



class SimpleCfg:
    def __init__(self):
        self.input_size = 3
        self.normalise_obs = True
        self.simulationTime = 150
cfg = SimpleCfg()

# Initialize environment with default arguments
env = envWrapper.LRRenv(
    inputFile=os.path.join(script_dir,"../../../example_problems/custom_warehouse.domain/warehouse_8x6.json"),
    outputFile=os.path.join(script_dir,"../../../outputs/pyTest.json"),
    simulationTime=cfg.simulationTime,
    planTimeLimit=70,
    preprocessTimeLimit=30000,
    observationTypes={"node-basics"},
    random_agents_and_tasks="true"
)
env.make_env_params_available()

parser = LRRParser(env, cfg)

number_of_runs = 1000

obs_vec = torch.zeros((number_of_runs*cfg.simulationTime, env.nNodes, cfg.input_size))
action_vec = torch.zeros((number_of_runs*cfg.simulationTime, env.nNodes))
next_index = 0

num_tasks_finished_sum = 0
for i in range(number_of_runs):

    print("\n\n\n\n")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print(f"\n       Run:{i+1} of {number_of_runs}\n")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("\n\n\n\n")

    
    # Reset environment with optional new parameters
    obs, reward, done = env.reset()
    step = 0
    obs_parsed = parser.parse_obs(obs)


    while not done:
        action_rl = skip_actor(env, obs)
        total_agents = np.sum(obs["free_agents_per_node"])
        desired_agent_dist = assign_discrete_actions(total_agents, action_rl)

        if total_agents > 0:
            obs_vec[next_index] = obs_parsed.x
            action_vec[next_index] = torch.tensor(action_rl)
            next_index += 1

        reb_action = solveRebFlow(
            env,
            obs,
            desired_agent_dist,
            "None",
        )
        action_dict = {"reb_action": reb_action}

        obs, reward_dict, done, info = env.step(action_dict)

        obs_parsed = parser.parse_obs(obs)
        num_tasks_finished_sum += reward_dict["task-finished"]
        step += 1

obs_vec = obs_vec[:next_index]
action_vec = action_vec[:next_index]
edge_index = obs_parsed.edge_index

dataset_dict = {
    "obs": obs_vec,
    "actions": action_vec,
    "edge_index": edge_index,
    "nAgents": env.nAgents,
    "normalise_obs": cfg.normalise_obs,
}

torch.save(dataset_dict, os.path.join(script_dir, "../../../outputs/skip_dataset_normalized1000.pt"))
print("Dataset saved successfully.")






        


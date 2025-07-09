import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(script_dir, "build")
sys.path.append(build_path)

import envWrapper

# Initialize environment with default arguments
env = envWrapper.LRRenv(
    inputFile="./example_problems/custom_warehouse.domain/warehouse_8x6.json",
    outputFile="./outputs/pyTest.json",
    simulationTime=150,
    planTimeLimit=70,
    preprocessTimeLimit=30000,
    observationTypes={"node-basics"},    
    random_agents_and_tasks="true",
    scheduler_type="ILP",    # ActivatedGreedy, ActivatedAdvantage, NoManSky, default, ILP, GreedyOptiDist, ILPOptiDist
    planner_type="default",
    guarantee_planner_time = True
)
env.make_env_params_available()


number_of_runs = 10

sum_reward = 0
sum_Astar_reward = 0
for i in range(number_of_runs):
    this_reward = 0
    Astar_reward = 0
    # Reset environment with optional new parameters
    obs, reward, done = env.reset()

    print("RESET PYTHON")
    print(f"Reward: {reward}, Done: {done}")
    print(f"Free agents: {obs["free_agents_per_node"]}")
    print(f"Free tasks: {obs["free_tasks_per_node"]}")
    print(f"nNodes: {env.nNodes}, nAgents: {env.nAgents}, nTasks: {env.nTasks}, nRoadmapNodes: {env.nRoadmapNodes}")


    while not done:
        # Take a step in the environment
        # action_dict = {"roadmap_activation": [1] * env.nRoadmapNodes}
        # action_dict = {"activation_action": np.ones((env.nNodes, 2), dtype=int)}
        obs, reward, done, info = env.step()
        this_reward += reward["task-finished"]
        Astar_reward += reward["A*-distance"]
        print(f"Astar reward: {reward['A*-distance']}, Task reward: {reward['task-finished']}, Idle agents reward: {reward['idle-agents']}")

    print("One simulation complete with reward: ", this_reward)
    print("avergae reward: ", this_reward / env.nTasks)
    print("Astar reward: ", Astar_reward)
    sum_reward += this_reward
    sum_Astar_reward += Astar_reward
    print(f"Average reward after {i+1} runs: {sum_reward/(i+1)}")

print("Average reward over ", number_of_runs, " runs: ", sum_reward/number_of_runs)
print("Average Astar reward over ", number_of_runs, " runs: ", sum_Astar_reward/number_of_runs)
print("Simulation complete.")

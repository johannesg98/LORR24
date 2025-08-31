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
    simulationTime=10000,
    planTimeLimit=100,
    preprocessTimeLimit=30000,
    observationTypes={"node-basics"},    
    random_agents_and_tasks="true",
    scheduler_type="default",    # ActivatedGreedy, ActivatedAdvantage, NoManSky, default, ILP, GreedyOptiDist, ILPOptiDist
    planner_type="default",
    guarantee_planner_time = True
)
env.make_env_params_available()


number_of_runs = 1

sum_reward = 0
sum_Astar_reward = 0
sum_episode_time_in_task = 0
sum_length_of_tasks_finished = 0
sum_wait_time = 0
sum_n_best_pibt_step = 0
sum_n_not_best_pibt_step = 0
for i in range(number_of_runs):
    this_reward = 0
    Astar_reward = 0
    episode_time_in_task = 0
    episode_length_of_tasks_finished = 0
    episode_wait_time = 0
    episode_n_best_pibt_step = 0
    episode_n_not_best_pibt_step = 0
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
        episode_time_in_task += info["agents-in-task"]
        episode_length_of_tasks_finished += info["length-of-tasks-finished"]
        episode_wait_time += info["wait-time"]
        episode_n_best_pibt_step += info["n-best-pibt-step"]
        episode_n_not_best_pibt_step += info["n-not-best-pibt-step"]
        print(f"Astar reward: {reward['A*-distance']}, Task reward: {reward['task-finished']}, Idle agents reward: {reward['idle-agents']}")
        if done:
            np.save(os.path.join(script_dir, f"../outputs/pibt_wait_map_test.npy"), np.array(info["pibt-wait-map"]))

    print("One simulation complete with reward: ", this_reward)
    print("avergae reward: ", this_reward / env.nTasks)
    print("Astar reward: ", Astar_reward)
    print("Time in task: ", episode_time_in_task)
    sum_reward += this_reward
    sum_Astar_reward += Astar_reward
    sum_episode_time_in_task += episode_time_in_task
    sum_length_of_tasks_finished += episode_length_of_tasks_finished
    sum_wait_time += episode_wait_time
    sum_n_best_pibt_step += episode_n_best_pibt_step
    sum_n_not_best_pibt_step += episode_n_not_best_pibt_step
    print(f"Average reward after {i+1} runs: {sum_reward/(i+1)}")
    print(f"Average Time in task after {i+1} runs: {sum_episode_time_in_task/(i+1)}")
    print(f"Average Length of tasks finished after {i+1} runs: {sum_length_of_tasks_finished/(i+1)}")
    print(f"Average Wait time after {i+1} runs: {sum_wait_time/(i+1)}")
    print(f"Average n_best_pibt_step after {i+1} runs: {sum_n_best_pibt_step/(i+1)}")
    print(f"Average n_not_best_pibt_step after {i+1} runs: {sum_n_not_best_pibt_step/(i+1)}")


print("Average reward over ", number_of_runs, " runs: ", sum_reward/number_of_runs)
print("Average Astar reward over ", number_of_runs, " runs: ", sum_Astar_reward/number_of_runs)
print("Simulation complete.")

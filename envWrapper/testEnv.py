import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(script_dir, "build")
sys.path.append(build_path)

import envWrapper

# Initialize environment with default arguments
env = envWrapper.LRRenv(
    inputFile="./example_problems/custom_warehouse.domain/warehouse_4x3_100.json",
    outputFile="./outputs/pyTest.json",
    simulationTime=100,
    planTimeLimit=40,
    preprocessTimeLimit=30000,
    observationTypes={"node-basics"}
)
env.make_env_params_available()


number_of_runs = 10

sum_reward = 0
for i in range(number_of_runs):
    this_reward = 0
    # Reset environment with optional new parameters
    obs, reward, done = env.reset()
    # env.step()
    # env.step()
    # env.step()
    # env.step()
    # env.step()
    # obs, reward, done = env.reset()

    print("RESET PYTHON")
    print(f"Reward: {reward}, Done: {done}")
    print(f"Free agents: {obs["free_agents_per_node"]}")
    print(f"Free tasks: {obs["free_tasks_per_node"]}")
    print(f"nNodes: {env.nNodes}, nAgents: {env.nAgents}, nTasks: {env.nTasks}")


    #//obs = env.get_observation().to_py_dict()
    #oder: obs = env.get_observation() aber kp muss eh aus reset rausgeholt werden

    while not done:
        # Take a step in the environment
        obs, reward, done = env.step()
        this_reward += reward

    print("One simulation complete with reward: ", this_reward)
    sum_reward += this_reward

print("Average reward over ", number_of_runs, " runs: ", sum_reward/number_of_runs)
print("Simulation complete.")

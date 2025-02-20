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
    simulationTime=25,
    planTimeLimit=300,
    preprocessTimeLimit=30000,
    observationTypes={"node-basics"}
)
env.make_env_params_available()
print(f"VOR RESET nNodes: {env.nNodes}, nAgents: {env.nAgents}, nTasks: {env.nTasks}")
print("Node Cost Matrix: ")
print(env.NodeCostMatrix)

# Reset environment with optional new parameters
obs, reward, done = env.reset()

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
    print("STEP PYTHON")
    print(f"Reward: {reward}, Done: {done}")
    print(f"Free agents: {obs["free_agents_per_node"]}")
    print(f"Free tasks: {obs["free_tasks_per_node"]}")
    print(f"Type of free agents: {type(obs["free_agents_per_node"])}")

print("Simulation complete.")

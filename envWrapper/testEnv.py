import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(script_dir, "build")
sys.path.append(build_path)

import envWrapper

# Initialize environment with default arguments
env = envWrapper.LRRenv(
    inputFile="./example_problems/random.domain/random_32_32_20_200.json",
    outputFile="./outputs/pyTest.json",
    simulationTime=10,
    planTimeLimit=300,
    preprocessTimeLimit=30000
)


# Reset environment with optional new parameters
env.reset()

print("reset env done py\n")

# Step through the environment
done = False
while not done:
    # Take a step in the environment
    state, reward, done = env.step()
    print(f"State: {state}, Reward: {reward}, Done: {done}")

print("Simulation complete.")

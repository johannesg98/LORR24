
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, value
import pulp
import numpy as np




def skip_actor(env, obs):

    edges = [(i, j) for i in range(env.nNodes) for j in range(env.nNodes)]

    # Define the PuLP problem
    model = LpProblem("BestTaskAssignment", LpMinimize)

    # Decision variables: rebalancing flow on each edge
    rebFlow = {(i, j): LpVariable(f"rebFlow_{i}_{j}", lowBound=0, cat='Integer') for (i, j) in edges}

    # Objective: minimize total distance (cost) of rebalancing flows
    model += lpSum(rebFlow[(i, j)] * env.NodeCostMatrix[i][j] for (i, j) in edges), "TotalRebalanceCost"

    # Constraints for each region (node)
    for k in range(env.nNodes):
        # Assign all agents to tasks
        model += (
            lpSum(rebFlow[(k, j)] for j in range(env.nNodes)) == obs["free_agents_per_node"][k], 
            f"RebalanceSupply_{k}"
        )

        # Limit tasks per node to available tasks
        model += (
            lpSum(rebFlow[(i, k)] for i in range(env.nNodes)) <= obs["free_tasks_per_node"][k], 
            f"TaskDemand_{k}"
        )

    # Solve the problem
    status = model.solve(pulp.PULP_CBC_CMD(msg=False, options=["primalTol=1e-9", "dualTol=1e-9", "mipGap=1e-9"]))

    # Check if the solution is optimal
    if LpStatus[status] == "Optimal":
        assined_per_node = [0] * env.nNodes
        for (i, j) in edges:
            assined_per_node[j] += int(rebFlow[(i, j)].varValue)

        action_rl = np.array([assined_per_node])/np.sum(assined_per_node)
        
        return action_rl
    else:
        print(f"Skip actor optimization failed with status: {LpStatus[status]}")
        return None
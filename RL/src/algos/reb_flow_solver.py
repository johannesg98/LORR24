import os
import subprocess
from collections import defaultdict
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, value
import pulp

def solveRebFlow(env, obs, desired_agent_dist, CPLEXPATH):
    if CPLEXPATH=='None':
        return solveRebFlow_pulp(env, obs, desired_agent_dist)
    else: 
        print("ERROR: CPLEX solver not implemented.")

def solveRebFlow_pulp(env, obs, desired_agent_dist):



    # Fully connected graph
    edges = [(i, j) for i in range(env.nNodes) for j in range(env.nNodes) if i!=j]

    # Define the PuLP problem
    model = LpProblem("RebalancingFlowMinimization", LpMinimize)

    # Decision variables: rebalancing flow on each edge
    rebFlow = {(i, j): LpVariable(f"rebFlow_{i}_{j}", lowBound=0, cat='Integer') for (i, j) in edges}

    # Objective: minimize total distance (cost) of rebalancing flows
    model += lpSum(rebFlow[(i, j)] * env.NodeCostMatrix[i][j] for (i, j) in edges), "TotalRebalanceCost"

    # Constraints for each region (node)
    for k in range(env.nNodes):
        # 1. Flow conservation constraint (ensure net inflow/outflow achieves desired vehicle distribution)
        model += (
            lpSum(rebFlow[(j, i)]-rebFlow[(i, j)] for (i, j) in edges if j != i and i==k)
        ) == desired_agent_dist[k] - obs["free_agents_per_node"][k], f"FlowConservation_{k}"

        # 2. Rebalancing flows from region i should not exceed the available vehicles in region i
        model += (
            lpSum(rebFlow[(i, j)] for (i, j) in edges if i != j and i==k) <= obs["free_agents_per_node"][k], 
            f"RebalanceSupply_{k}"
        )

    # Solve the problem
    status = model.solve(pulp.PULP_CBC_CMD(msg=False, options=["primalTol=1e-9", "dualTol=1e-9", "mipGap=1e-9"]))

    # Check if the solution is optimal
    if LpStatus[status] == "Optimal":
        # Collect the rebalancing flows
        flow = defaultdict(int)
        outgoing_per_node = [0] * env.nNodes
        for (i, j) in edges:
            flow[(i, j)] = int(rebFlow[(i, j)].varValue)
            outgoing_per_node[i] += flow[(i, j)]
        #add all agents that stay at a node
        for i in range(env.nNodes):
            flow[(i, i)] = obs["free_agents_per_node"][i] - outgoing_per_node[i]
        #print(len(rebFlow.keys()))
       
        #flow_result = {(i, j): value(rebFlow[(i, j)]) for (i, j) in edges}
        
        #action = [flow[i,j] for i,j in env.edges]
        return flow
    else:
        print(f"Optimization failed with status: {LpStatus[status]}")
        return None



   
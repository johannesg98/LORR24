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
    edges = [(i, j) for i in range(env.nNodes) for j in range(env.nNodes)]

    # Define the PuLP problem
    model = LpProblem("RebalancingFlowMinimization", LpMinimize)

    # Decision variables: rebalancing flow on each edge
    rebFlow = {(i, j): LpVariable(f"rebFlow_{i}_{j}", lowBound=0, cat='Integer') for (i, j) in edges}

    # Objective: minimize total distance (cost) of rebalancing flows
    model += lpSum(rebFlow[(i, j)] * env.NodeCostMatrix[i][j] for (i, j) in edges), "TotalRebalanceCost"


    ##CONTINUE FROM HERE
    ##XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    ##XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


    # Constraints for each region (node)
    for k in range(env.nNodes):
        # 1. Flow conservation constraint (ensure net inflow/outflow achieves desired vehicle distribution)
        model += (
            lpSum(rebFlow[(j, i)]-rebFlow[(i, j)] for (i, j) in edges if j != i and i==k)
        ) >= desired_vehicles[k] - acc_init[k], f"FlowConservation_{k}"

        # 2. Rebalancing flows from region i should not exceed the available vehicles in region i
        model += (
            lpSum(rebFlow[(i, j)] for (i, j) in edges if i != j and i==k) <= acc_init[k], 
            f"RebalanceSupply_{k}"
        )







    t = env.time
    
    # Prepare the data: rounding desiredAcc and getting current vehicle counts
    accRLTuple = [(n, int(round(desiredAcc[n]))) for n in desiredAcc]
    accTuple = [(n, int(env.acc[n][t+1])) for n in env.acc]
    
    # Extract the edges and the times
    edgeAttr = [(i, j, env.G.edges[i, j]['time']) for i, j in env.G.edges]
    edges = [(i, j) for i, j in env.G.edges]

    # Map vehicle availability and desired vehicles for each region
    acc_init = {n: int(env.acc[n][t+1]) for n in env.acc}
    desired_vehicles = {n: int(round(desiredAcc[n])) for n in desiredAcc}

    region = [n for n in acc_init]
    # Time on each edge (used in the objective)
    time = {(i, j): env.G.edges[i, j]['time'] for i, j in edges}

    # Define the PuLP problem
    model = LpProblem("RebalancingFlowMinimization", LpMinimize)
    
    # Decision variables: rebalancing flow on each edge
    rebFlow = {(i, j): LpVariable(f"rebFlow_{i}_{j}", lowBound=0, cat='Integer') for (i, j) in edges}

    # Objective: minimize total time (cost) of rebalancing flows
    model += lpSum(rebFlow[(i, j)] * time[(i, j)] for (i, j) in edges), "TotalRebalanceCost"
    
    # Constraints for each region (node)
    for k in region:
        # 1. Flow conservation constraint (ensure net inflow/outflow achieves desired vehicle distribution)
        model += (
            lpSum(rebFlow[(j, i)]-rebFlow[(i, j)] for (i, j) in edges if j != i and i==k)
        ) >= desired_vehicles[k] - acc_init[k], f"FlowConservation_{k}"

        # 2. Rebalancing flows from region i should not exceed the available vehicles in region i
        model += (
            lpSum(rebFlow[(i, j)] for (i, j) in edges if i != j and i==k) <= acc_init[k], 
            f"RebalanceSupply_{k}"
        )
    
    # Solve the problem
    status = model.solve(pulp.PULP_CBC_CMD(msg=False, options=["primalTol=1e-9", "dualTol=1e-9", "mipGap=1e-9"]))
    #print objective value 
    # Check if the solution is optimal
    if LpStatus[status] == "Optimal":
        # Collect the rebalancing flows
        flow = defaultdict(float)
        for (i, j) in edges:
            flow[(i, j)] = rebFlow[(i, j)].varValue
        #print(len(rebFlow.keys()))
       
        #flow_result = {(i, j): value(rebFlow[(i, j)]) for (i, j) in edges}
        
        action = [flow[i,j] for i,j in env.edges]
        return action
    else:
        print(f"Optimization failed with status: {LpStatus[status]}")
        return None


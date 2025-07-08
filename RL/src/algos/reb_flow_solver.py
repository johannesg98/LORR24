import os
import subprocess
from collections import defaultdict
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, value
import pulp
import time
import numpy as np

# try:
#     import cplex
#     CPLEX_AVAILABLE = True
# except ImportError:
#     CPLEX_AVAILABLE = False

def solveRebFlow(env, obs, desired_agent_dist, CPLEXPATH):
    # if CPLEX_AVAILABLE:
    #     return solveRebFlow_cplex(env, obs, desired_agent_dist)
    # else: 
    return solveRebFlow_pulp(env, obs, desired_agent_dist)
        

def solveRebFlow_pulp(env, obs, desired_agent_dist):

    NodeCostMatrix = env.NodeCostMatrix
    nNodes = env.nNodes




    # Fully connected graph
    # edges = [(i, j) for i in range(env.nNodes) for j in range(env.nNodes) if i!=j]
    edges = [(i, j) for i in range(nNodes) for j in range(nNodes) if i!=j and ((obs["free_agents_per_node"][i] > 0 and desired_agent_dist[j] > 0) or (obs["free_agents_per_node"][j] > 0 and desired_agent_dist[i] > 0))]

    print("Blub 0")

    # Define the PuLP problem
    model = LpProblem("RebalancingFlowMinimization", LpMinimize)
  
    # Decision variables: rebalancing flow on each edge
    # rebFlow = {(i, j): LpVariable(f"rebFlow_{i}_{j}", lowBound=0, cat='Integer') for (i, j) in edges}
    rebFlow = {(i, j): LpVariable(f"rebFlow_{i}_{j}", lowBound=0, cat='Continuous') for (i, j) in edges}

    
    print("Blub 1")
   
    # Objective: minimize total distance (cost) of rebalancing flows
    model += lpSum(rebFlow[(i, j)] * NodeCostMatrix[i][j] for (i, j) in edges), "TotalRebalanceCost"

    
    print("Blub 2")
    
    # Constraints for each region (node)
    for k in range(nNodes):
        # 1. Flow conservation constraint (ensure net inflow/outflow achieves desired vehicle distribution)
        model += (
            lpSum(rebFlow[(j, i)]-rebFlow[(i, j)] for (i, j) in edges if j != i and i==k)
        ) == desired_agent_dist[k] - obs["free_agents_per_node"][k], f"FlowConservation_{k}"

        # 2. Rebalancing flows from region i should not exceed the available vehicles in region i
        model += (
            lpSum(rebFlow[(i, j)] for (i, j) in edges if i != j and i==k) <= obs["free_agents_per_node"][k], 
            f"RebalanceSupply_{k}"
        )

    
    print("Blub 3")

    # Solve the problem
    status = model.solve(pulp.PULP_CBC_CMD(msg=False, options=["primalTol=1e-9", "dualTol=1e-9", "mipGap=1e-9"]))

    
    print("Blub 4")

    # Check if the solution is optimal
    if LpStatus[status] == "Optimal":
        # # Collect the rebalancing flows
        # flow = defaultdict(int)
        # outgoing_per_node = [0] * nNodes
        # for (i, j) in edges:
        #     flow[(i, j)] = int(rebFlow[(i, j)].varValue)
        #     outgoing_per_node[i] += flow[(i, j)]
        # #add all agents that stay at a node
        # for i in range(nNodes):
        #     flow[(i, i)] = obs["free_agents_per_node"][i] - outgoing_per_node[i]
        # #add edges that are not in edges (and therefore 0 by default)
        # for i in range(nNodes):
        #     for j in range(nNodes):
        #         if (i,j) not in edges and i!=j:
        #             flow[(i,j)] = 0
        #print(len(rebFlow.keys()))
       
        #flow_result = {(i, j): value(rebFlow[(i, j)]) for (i, j) in edges}
        
        #action = [flow[i,j] for i,j in env.edges]


        # Collect the rebalancing flows
        flow = np.zeros((nNodes, nNodes), dtype=int)
        outgoing_per_node = np.zeros(nNodes, dtype=int)
        for (i, j) in edges:
            flow[i,j] = int(rebFlow[(i, j)].varValue)
            outgoing_per_node[i] += flow[i, j]
        #add all agents that stay at a node
        np.fill_diagonal(flow, np.array(obs["free_agents_per_node"]) - outgoing_per_node)
        #add edges that are not in edges (and therefore 0 by default)
        # - not necessary

        print("Blub 5")

        return flow
    else:
        print(f"Optimization failed with status: {LpStatus[status]}")
        return None



def solveRebFlow_cplex(env, obs, desired_agent_dist, CPLEXPATH=None):
    """
    Solve rebalancing flow optimization using CPLEX.
    Uses matrix-based constraint construction for efficiency.
    """
    start = time.time()
    
    if CPLEXPATH and CPLEXPATH != 'None':
        # Set CPLEX path if provided
        os.environ['CPLEX_STUDIO_DIR1210'] = CPLEXPATH
    
    NodeCostMatrix = env.NodeCostMatrix
    nNodes = env.nNodes
    
    print("CPLEX YEHAAA nNodes:", nNodes)
    
    # Create edges - same logic as PuLP version
    edges = [(i, j) for i in range(nNodes) for j in range(nNodes) 
             if i != j and ((obs["free_agents_per_node"][i] > 0 and desired_agent_dist[j] > 0) or 
                           (obs["free_agents_per_node"][j] > 0 and desired_agent_dist[i] > 0))]
    
    # Create a mapping from edge tuples to variable indices
    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
    n_edges = len(edges)
    
    try:
        # Create CPLEX problem
        model = cplex.Cplex()
        model.set_problem_type(cplex.Cplex.problem_type.MILP)
        model.objective.set_sense(model.objective.sense.minimize)
        
        # Suppress CPLEX output
        model.set_log_stream(None)
        model.set_error_stream(None)
        model.set_warning_stream(None)
        model.set_results_stream(None)
        
        # Add decision variables: rebalancing flow on each edge
        var_names = [f"rebFlow_{i}_{j}" for i, j in edges]
        var_types = [model.variables.type.integer] * n_edges
        var_lb = [0.0] * n_edges  # Lower bounds
        var_ub = [cplex.infinity] * n_edges  # Upper bounds
          # Objective coefficients (costs)
        obj_coeffs = [float(NodeCostMatrix[i][j]) for i, j in edges]
        
        model.variables.add(obj=obj_coeffs, lb=var_lb, ub=var_ub, types=var_types, names=var_names)

        print("Cplex setup time:", time.time() - start)
        startTmp = time.time()
        
        # Build constraints efficiently using matrix approach
        constraint_names = []
        constraint_senses = []
        constraint_rhs = []
        constraint_rows = []
          # 1. Flow conservation constraints
        for k in range(nNodes):
            # Net flow = inflow - outflow = desired - current
            row_indices = []
            row_coeffs = []
            
            for edge_idx, (i, j) in enumerate(edges):
                if j == k:  # Inflow to node k
                    row_indices.append(edge_idx)
                    row_coeffs.append(1.0)
                elif i == k:  # Outflow from node k
                    row_indices.append(edge_idx)
                    row_coeffs.append(-1.0)
            
            if row_indices:  # Only add constraint if there are relevant edges
                constraint_names.append(f"FlowConservation_{k}")
                constraint_senses.append('E')  # Equality
                constraint_rhs.append(float(desired_agent_dist[k] - obs["free_agents_per_node"][k]))
                constraint_rows.append([row_indices, row_coeffs])
          # 2. Supply constraints (outflow <= available vehicles)
        for k in range(nNodes):
            row_indices = []
            row_coeffs = []
            
            for edge_idx, (i, j) in enumerate(edges):
                if i == k:  # Outflow from node k
                    row_indices.append(edge_idx)
                    row_coeffs.append(1.0)
            
            if row_indices:  # Only add constraint if there are outgoing edges
                constraint_names.append(f"RebalanceSupply_{k}")
                constraint_senses.append('L')  # Less than or equal
                constraint_rhs.append(float(obs["free_agents_per_node"][k]))
                constraint_rows.append([row_indices, row_coeffs])
        
        # Add constraints to model
        model.linear_constraints.add(
            lin_expr=constraint_rows,
            senses=constraint_senses,
            rhs=constraint_rhs,
            names=constraint_names
        )
        
        # Set solver parameters for precision
        model.parameters.mip.tolerances.mipgap.set(1e-9)
        model.parameters.mip.tolerances.absmipgap.set(1e-9)
        model.parameters.simplex.tolerances.feasibility.set(1e-9)
        model.parameters.simplex.tolerances.optimality.set(1e-9)

        print("Cplex fill time:", time.time() - startTmp)
        startTmp = time.time()
        
        # Solve the problem
        model.solve()

        print("Cplex solve time:", time.time() - startTmp)
        
        # Check solution status
        status = model.solution.get_status()
        if status == model.solution.status.MIP_optimal:
            # Get solution values
            solution_values = model.solution.get_values()
            
            # Build flow dictionary
            flow = defaultdict(int)
            outgoing_per_node = [0] * nNodes
            
            for edge_idx, (i, j) in enumerate(edges):
                flow_value = int(round(solution_values[edge_idx]))
                flow[(i, j)] = flow_value
                outgoing_per_node[i] += flow_value
            
            # Add agents that stay at each node
            for i in range(nNodes):
                flow[(i, i)] = obs["free_agents_per_node"][i] - outgoing_per_node[i]
            
            # Add edges not in the edge list (set to 0)
            for i in range(nNodes):
                for j in range(nNodes):
                    if (i, j) not in edges and i != j:
                        flow[(i, j)] = 0
            
            print("CPLEX total time:", time.time() - start)

            return flow
            
        else:
            print(f"CPLEX optimization failed with status: {status}")
            return None
            
    except Exception as e:
        print(f"CPLEX error: {e}")
        print("Falling back to PuLP solver...")
        return solveRebFlow_pulp(env, obs, desired_agent_dist)


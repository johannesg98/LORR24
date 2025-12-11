import os
import subprocess
from collections import defaultdict
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, value
import pulp
import time
import numpy as np

from ortools.graph.python import min_cost_flow


def solveRebFlow(env, obs, desired_agent_dist):
    NodeCostMatrix = env.NodeCostMatrix
    n = env.nNodes
    a = obs["free_agents_per_node"]
    d = desired_agent_dist

    mcf = min_cost_flow.SimpleMinCostFlow()

    # Indexing: 0..n-1 = source layer, n..2n-1 = sink layer
    def src(i): return i
    def snk(i): return n + i

    # Arcs from every source-node to every sink-node (including i->i for "stay")
    for i in range(n):
        for j in range(n):
            cost = int(NodeCostMatrix[i][j]) if i != j else 0  # 0 cost to stay
            # Capacity: at most a[i] can leave i, so this cap is safe/tight
            cap = int(a[i])
            if cap > 0 and d[j] > 0:  # Only add if there's supply and demand
                mcf.add_arc_with_capacity_and_unit_cost(src(i), snk(j), cap, cost)

    # Supplies on source-layer, demands on sink-layer
    for i in range(n):
        mcf.set_node_supply(src(i), int(a[i]))         # supply = initial agents
        mcf.set_node_supply(snk(i), -int(d[i]))        # demand = desired agents

    status = mcf.solve()
    if status != mcf.OPTIMAL:
        raise RuntimeError(f"MCF failed: {status}")

    # Recover n x n flow matrix (actual moves i->j)
    flow = np.zeros((n, n), dtype=int)
    for k in range(mcf.num_arcs()):
        u = mcf.tail(k); v = mcf.head(k); f = mcf.flow(k)
        if f == 0: continue
        if 0 <= u < n and n <= v < 2*n: #check if arc is from source to sink (technically not necessary)
            i = u
            j = v - n
            flow[i, j] += f

    return flow


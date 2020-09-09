import networkx as nx
from pedestrians import read_agentlist
from edges import Edge
import sys
import pandas as pd

pedestrians = read_agentlist("data/agentlist.txt")
edges = Edge("data")
# print(pedestrians)
# print(edges)
# print(edges.DistanceMatrix)
# print(edges.goal_capa)
# print(edges.observed_goal)

N = 5000
# N = 500
G = nx.DiGraph()
G.add_node("O", demand=-N)
G.add_node("D", demand=N)

for node_id, goal_id in edges.observed_goal.items():
    G.add_node("G%d"%goal_id, demand=0)
    G.add_edge("G%d"%goal_id, "D", weight=0, capacity=edges.goal_capa[goal_id])


for ped in pedestrians[:N]:
    G.add_node("P%d"%ped.idx, demand=0)
    G.add_edge("O", "P%d"%ped.idx, weight=0, capacity=1)
    for node_id, goal_id in edges.observed_goal.items():
        dist = edges.DistanceMatrix[edges.observed_goal[ped.destination], goal_id]
        w = int(dist/ped.speed)
        # print(ped.idx, ped.destination, goal_id, dist, w)
        G.add_edge("P%d"%ped.idx, "G%d"%goal_id, weight=int(dist/ped.speed), capacity=1)

# sys.exit()

# G.add_edge("a", "b", weight=3, capacity=4)
# G.add_edge("a", "c", weight=6, capacity=10)
# G.add_edge("b", "d", weight=1, capacity=9)
# G.add_edge("c", "d", weight=2, capacity=5)
flowCost, flowDict = nx.capacity_scaling(G, demand="demand", capacity="capacity", weight="weight")
for k,v in flowDict.items():
    if "P" not in k:
        continue
    for k2,v2 in v.items():
        if v2 == 0:
            continue
        print(k, k2, v2)
# print(flowDict)
print(flowCost)
pd.to_pickle(flowDict, "flowDict.pickle")
pd.to_pickle(flowCost, "flowCost.pickle")

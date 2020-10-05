import networkx as nx
from pedestrians import read_agentlist
from edges import Edge
import sys, os
import pandas as pd
from collections import defaultdict

pedestrians = read_agentlist("data/agentlist.txt")
edges = Edge("data")
# print(pedestrians)
# print(edges)
# print(edges.DistanceMatrix)
# print(edges.goal_capa)
# print(edges.observed_goal)

N = 5000
# N = 500

def calc_save_min_cost_flow():
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

    flowCost, flowDict = nx.capacity_scaling(G, demand="demand", capacity="capacity", weight="weight")
    pd.to_pickle(flowDict, "flowDict.pickle")
    pd.to_pickle(flowCost, "flowCost.pickle")

def calc_save_min_cost_flow2():
    G = nx.DiGraph()
    G.add_node("O", demand=-N)
    G.add_node("D", demand=N)

    tmp_S = {}
    for node_id, goal_id in edges.observed_goal.items():
        G.add_node("G%d"%goal_id, demand=0)
        G.add_edge("G%d"%goal_id, "D", weight=0, capacity=edges.goal_capa[goal_id])
        tmp_S[goal_id] = 0

    for ped in pedestrians[:N]:
        # G.add_node("P%d"%ped.idx, demand=0)
        # G.add_edge("O", "P%d"%ped.idx, weight=0, capacity=1)
        tmp_S[edges.observed_goal[ped.destination]] += 1
        # for node_id, goal_id in edges.observed_goal.items():
        #     dist = edges.DistanceMatrix[edges.observed_goal[ped.destination], goal_id]
        #     w = int(dist/ped.speed)
        #     print(ped.idx, ped.destination, goal_id, dist, w)
        #     G.add_edge("P%d"%ped.idx, "G%d"%goal_id, weight=int(dist/ped.speed), capacity=1)


    for node_id, goal_id in edges.observed_goal.items():
        # スタートする避難所=歩行者の当初の目的地
        G.add_node("S%d"%goal_id, demand=0)
        G.add_edge("O", "S%d"%goal_id, weight=0, capacity=tmp_S[goal_id])
        for node_id2, goal_id2 in edges.observed_goal.items():
            dist = edges.DistanceMatrix[edges.observed_goal[goal_id, goal_id2]
            G.add_edge("S%d"%goal_id, "G%d"%goal_id2, weight=int(dist), capacity=tmp_)

    
    flowCost, flowDict = nx.capacity_scaling(G, demand="demand", capacity="capacity", weight="weight")
    pd.to_pickle(flowDict, "flowDict.pickle")
    pd.to_pickle(flowCost, "flowCost.pickle")


if __name__=="__main__":
    if not os.path.exists("flowDict.pickle"):
        calc_save_min_cost_flow()
    
    flowDict = pd.read_pickle("flowDict.pickle")
    flowCost = pd.read_pickle("flowCost.pickle")
    dict_cnt = defaultdict(int)
    for k,v in flowDict.items():
        if "P" not in k:
            continue
        for k2,v2 in v.items():
            if v2 == 0:
                continue
            dict_cnt[k2] += 1
            # print(k, k2, v2)
    # print(edges.goal_capa)
    # for goalid, capa in enumerate(edges.goal_capa):
    for goalid, sid in edges.observed_goal.items():
        capa = edges.goal_capa[sid]
        goal_label = "G%d"%sid
        print(goalid, goal_label, dict_cnt[goal_label], capa)
    # print(dict_cnt)
    # print(flowDict)
    print(flowCost / 5000)
    # print(edges.observed_goal)

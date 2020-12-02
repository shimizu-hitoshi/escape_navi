#!/usr/bin/env python

import networkx as nx
import sys, os
import pandas as pd
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pedestrians import read_agentlist
from pedestrians import save_agentlist
from edges import Edge

pedestrians = read_agentlist("../data/agentlist.txt")
edges = Edge("../data")
# print(pedestrians)
# print(edges)
# print(edges.DistanceMatrix)
# print(edges.goal_capa)
# print(edges.observed_goal)

N = 5000
# N = 500

def convert_agentlist():
    """
    移動速度を考慮して，移動時間の総和を最小化する，という結果が計算済みの前提
    """
    flowDict = pd.read_pickle("flowDict_person.pickle")
    flowCost = pd.read_pickle("flowCost_person.pickle")
    # dict_cnt = defaultdict(int)

    # イベントファイル出力用
    # cmd_name = "signage"
    # ratio = 1.0
    # via_length = 0
    # tmp = [t, cmd_name, shelter, shelter, target_node, target_node, ratio, via_length]
    goal_ids = sorted( edges.observed_goal.keys() )
    t = 1
    # out.append( "%d signage %d %d %d %d 1.0 0"%(t, nid, idx, detour, detour) )
    fp = open("event.txt", "w")
    for k,v in flowDict.items():
        if "P" not in k:
            continue
        idx = int(k.split("P")[-1])
        # print(pedestrians[idx-1].idx)
        pedestrians[idx-1].group_idx = idx # 個別に誘導するためにグループを個別にする
        nid = pedestrians[idx-1].destination
        for k2,v2 in v.items():
            if v2 == 0:
                continue
            # print(k2, v2)
            break
        goal_id = int(k2.split("G")[-1])
        detour = goal_ids[goal_id]
        if detour == pedestrians[idx-1].destination:
            continue
        fp.write( "%d signage %d %d %d %d 1.0 0\n"%(t, nid, idx, detour, detour) )
    fp.close()
    save_agentlist("agentlist_new.txt", pedestrians)

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
    pd.to_pickle(flowDict, "flowDict_person.pickle")
    pd.to_pickle(flowCost, "flowCost_person.pickle")

def person_base_main():
    """
    移動速度を考慮して，移動時間の総和を最小化する
    """
    if not os.path.exists("flowDict_person.pickle"):
        calc_save_min_cost_flow()
    flowDict = pd.read_pickle("flowDict_person.pickle")
    flowCost = pd.read_pickle("flowCost_person.pickle")
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
    print(tmp_S)
    for node_id, goal_id in edges.observed_goal.items():
        # スタートする避難所=歩行者の当初の目的地
        G.add_node("S%d"%goal_id, demand=0)
        G.add_edge("O", "S%d"%goal_id, weight=0, capacity=tmp_S[goal_id])
        for node_id2, goal_id2 in edges.observed_goal.items():
            dist = edges.DistanceMatrix[goal_id, goal_id2]
            G.add_edge("S%d"%goal_id, "G%d"%goal_id2, weight=int(dist), capacity=N)
    print(G.edges.data())
    flowCost, flowDict = nx.capacity_scaling(G, demand="demand", capacity="capacity", weight="weight")
    pd.to_pickle(flowDict, "flowDict_shelter.pickle")
    pd.to_pickle(flowCost, "flowCost_shelter.pickle")

def shelter_base_main():
    """
    移動速度を無視して，移動距離の総和を最小化する
    """
    # if not os.path.exists("flowDict.pickle"):
    #     calc_save_min_cost_flow()
    calc_save_min_cost_flow2()
    # sys.exit()
    flowDict = pd.read_pickle("flowDict_shelter.pickle")
    flowCost = pd.read_pickle("flowCost_shelter.pickle")
    print(flowDict)
    print(flowCost)
    # sys.exit()

    dict_cnt = {} # defaultdict{defaultdict(int)}
    for k,v in flowDict.items():
        if "S" not in k:
            continue
        dict_cnt[k] = defaultdict(int)
        for k2,v2 in v.items():
            if v2 == 0:
                continue
            dict_cnt[k][k2] += v2
            # print(k, k2, v2)
    # print(edges.goal_capa)
    # for goalid, capa in enumerate(edges.goal_capa):
    for goalid, sid in edges.observed_goal.items():
        start_label = "S%d"%sid
        for goalid2, sid2 in edges.observed_goal.items():
            # capa = edges.goal_capa[sid2]
            goal_label = "G%d"%sid2
            if dict_cnt[start_label][goal_label] == 0: continue
            print(start_label, goal_label, dict_cnt[start_label][goal_label])
            # print(goalid, start_label, goal_label, dict_cnt[start_label][goal_label], capa)
    # print(dict_cnt)
    # print(flowDict)
    print(flowCost / 5000 / 1.2)
    # print(edges.observed_goal)

if __name__=="__main__":
    # person_base_main()
    shelter_base_main()
    # convert_agentlist()
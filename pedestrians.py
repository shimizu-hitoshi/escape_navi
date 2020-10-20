#!/usr/bin/env python

import numpy as np
import os, re
import networkx as nx
from edges import Edge

class Crowd():
    def __init__(self, ifn):
        self.pedestrians = read_agentlist(ifn)


class Pedestrian():
    def __init__(self, line):
        self.idx = int( line[0] )
        self.departure = int( line[1] )
        self.speed = float( line[2] )
        self.origin = int( line[3] )
        self.destination = int( line[4] )
        self.group_idx = int( line[5] ) # 基本的にゴールIDに揃える
        self.direction_idx = int( line[6] ) # ゴール後の電車の方向
        self.num_via = int( line[7] ) # 中継地点の数
        # self.vias = list( map(int, line[8:] ) ) # 中継地点のリストは単体テスト未実施

def read_agentlist(ifn):
    ret    = []
    with open(ifn) as fp:
        next(fp)
        for line in fp:
            line = line.strip().split()
            # line = line.strip().split("\t")
            ped = Pedestrian(line)
            ret.append(ped)
    return ret

def save_agentlist(ofn, agents):
    with open(ofn, "w") as fp:
        fp.write("%s 6\n"%len(agents))
        for agent in agents:
            out = "%d %d %.16f %d %d %d %d %d\n"%(
                agent.idx,
                agent.departure,
                agent.speed,
                agent.origin,
                agent.destination,
                agent.group_idx,
                agent.direction_idx,
                agent.num_via)
            fp.write(out)

def calc_fix_time(ddir):
    ifn = "%s/agentlist.txt"%ddir
    pedestrians = read_agentlist(ifn)
    edges = Edge(ddir)
    dict_distance = {}
    distances = [] # 出発地から目的地までの距離
    fix_time = [] # かかる時間の最小値
    # print(len(pedestrians))
    for ped in pedestrians:
        if (ped.origin, ped.destination) in dict_distance:
            distance = dict_distance[(ped.origin, ped.destination)]
        else:
            distance = nx.shortest_path_length(edges.G, ped.origin, ped.destination, weight='weight')
            dict_distance[(ped.origin, ped.destination)] = distance
        distances.append(distance)
        fix_time.append(distance / ped.speed)
    return fix_time, distances

def test_read_save(ddir):
    ifn = "%s/agentlist.txt"%ddir
    pedestrians = read_agentlist(ifn)
    save_agentlist("%s/agentlist_new.txt"%ddir, pedestrians)

if __name__ == '__main__':
    ddir = "data"
    # fix_time, distances = calc_fix_time(ddir)
    # # print(dict_distance)
    # print(np.mean(fix_time))
    test_read_save(ddir)

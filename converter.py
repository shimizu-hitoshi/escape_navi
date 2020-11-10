#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from pedestrians import Crowd
import sys
import numpy as np

class RewardMaker(object):
    def __init__(self):
        pass

    def set_agentfns(self, datadirs):
        self.crowds = []
        for datadir in datadirs:
            ifn = "%s/agentlist.txt"%datadir
            self.crowds.append(Crowd(ifn))

    def format_travel_time(self, travel_time):
        tmp_travel_time = dict(zip(agentid, travel_time))

        return dict_travel_time

    def set_R_base(self, R_base): # 実行に数分かかるので，要改善
        T_open, dict_travel_time = R_base
        self.R_base = torch.zeros(len(T_open), len( self.crowds[0].first_dest.items() ))
        group_agents = []
        for goalid, (nodeid, agentids) in enumerate( sorted( self.crowds[0].first_dest.items()) ):
            # print(goalid, (nodeid, agentids))
            print(goalid, len(agentids), np.mean( [dict_travel_time[agentid-1] for agentid in agentids] ) )
            group_agents.append(agentids)
        # print(dict_travel_time)

        for t, infos in enumerate(T_open):
            travel_time = infos['travel_time']
            agentid = infos['agentid']
            for goalid, group_agent in enumerate( group_agents):
                print(t, goalid, len([ aid for aid in agentid if aid in group_agent]) )
                self.R_base[t, goalid] = len([ aid for aid in agentid if aid in group_agent])
            # dict_travel_time = dict(zip(agentid, travel_time))
            # print(i, len(dict_travel_time))


        # print("set_R_base")
        # sys.exit()

        self.T_open = T_open
        self.dict_travel_time = dict_travel_time
        # print(T_open) # 今のところ全て0

    def info2reward(self, infos, training_target, t):
        """
        t: 時刻
        """
        # print(infos)
        # print(self.crowds[0].first_dest)
        # sys.exit()
        reward = torch.zeros((len(infos),1))
        for i, info in enumerate(infos):
            if "travel_time" not in info:
                continue
            travel_time = info['travel_time']
            agentid = info['agentid']
            for goalid, (nodeid, agentids) in enumerate( sorted( self.crowds[0].first_dest.items()) ):
                if goalid != training_target: # 学習対象の避難所が誘導した歩行者だけを考慮する
                    continue
                num_agent = len( [aid for aid in agentids if aid in agentid] ) 
                reward[i,0] = 1.0 * ( num_agent - self.R_base[t, goalid] ) / self.R_base[t, goalid] 
            # tmp_travel_time = dict(zip(agentid, travel_time))
            #     reward[i,0] = np.mean( [tmp_travel_time[agentid] for agentid in agentids] )

        # reward = None
        return reward


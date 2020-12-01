#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from pedestrians import Crowd
import sys, copy
import numpy as np
from edges import Edge

DEBUG = False

class RewardMaker(object):
    def __init__(self):
        pass

    def set_agentfns(self, datadirs):
        self.crowds = []
        for datadir in datadirs:
            ifn = "%s/agentlist.txt"%datadir
            self.crowds.append(Crowd(ifn))

    def set_edges(self, datadirs):
        self.edges = []
        for datadir in datadirs:
            # ifn = "%s/graph.twd"%datadir
            self.edges.append(Edge(datadir))


    # def format_travel_time(self, travel_time):
    #     dict_travel_time = dict(zip(agentid, travel_time))

    #     return dict_travel_time

    def set_R_base(self, R_base): # 実行に数分かかるので，要改善
        T_open, dict_travel_time = R_base
        # R_base: 歩行中＠最初の避難所
        # A_base: 到着済＠最後の避難所
        self.R_base = torch.zeros(len(T_open), len( self.crowds[0].first_dest.items() ))
        self.A_base = torch.zeros(len(T_open), len( self.crowds[0].first_dest.items() ))
        self.group_agents = [] # goalidごとに，agentidリストを保持
        for goalid, (nodeid, agentids) in enumerate( sorted( self.crowds[0].first_dest.items()) ):
            # print(goalid, (nodeid, agentids))
            print(goalid, len(agentids), np.mean( [dict_travel_time[agentid-1] for agentid in agentids] ) )
            self.group_agents.append(agentids)
        # print(dict_travel_time)

        self.len_group_agent = torch.zeros(len(self.group_agents))
        for goalid, group_agent in enumerate( self.group_agents ):
            self.len_group_agent[goalid] = len(group_agent)

        for t, infos in enumerate(T_open):
            # num_pedestrian = np.sum( infos["edge_state"] )
            goal_cnt = infos["goal_cnt"]
            self.A_base[t,:] = torch.tensor(goal_cnt)

            agentid, travel_time = infos['goal_time']
            # print("set_R_base", t, len(agentid), "人ゴール")
            # travel_time = infos['travel_time']
            # agentid = infos['agentid']
            # 当初目的地ごとに歩行者数を設定して
            for goalid, group_agent in enumerate( self.group_agents ):
                self.R_base[t,goalid] = len(group_agent)

            # ゴールした人数を減算する
            for aid in agentid:
                goalid = self.edges[0].nodeid2goalid( self.crowds[0].pedestrians[aid-1].destination )
                self.R_base[t, goalid] -= 1

            # for goalid, group_agent in enumerate( self.group_agents):
            #     print(t, goalid, len([ aid for aid in agentid if aid in group_agent]) )
            #     self.R_base[t, goalid] = len([ aid for aid in agentid if aid in group_agent])
            # dict_travel_time = dict(zip(agentid, travel_time))
            # print(i, len(dict_travel_time))
        print("R_base",self.R_base)
        print("A_base",self.A_base)
        # print("set_R_base")
        # sys.exit()

        self.T_open = T_open
        self.dict_travel_time = dict_travel_time
        # print(T_open) # 今のところ全て0

    def info2rewardWalk(self, infos, training_target=None, t=None):
        """
        歩行人数から作成する報酬
        t: 時刻
        """
        # print(infos)
        # print(self.crowds[0].first_dest)
        # sys.exit()
        # reward = torch.zeros((len(infos),1)) # 歩行人数から
        if training_target is None:
            reward1 = torch.zeros((len(infos),len(self.group_agents))) # 全エージェント分
        else:
            reward1 = torch.zeros((len(infos),1)) # 歩行人数から
        # reward2 = torch.zeros((len(infos),1)) # 収容人数から
        for i, info in enumerate(infos):
            num_agent = torch.zeros(len(self.group_agents))
            agentid, travel_time = info['goal_time']
            # print(t, len(agentid), "人ゴール")
            # 当初目的地ごとに歩行者数を設定して
            num_agent = copy.deepcopy(self.len_group_agent) # 当初目的地ごとの歩行者数
            # ゴールした人数を減算する
            for aid in agentid: # たぶんここが遅い
                goalid = self.edges[0].nodeid2goalid( self.crowds[0].pedestrians[aid-1].destination )
                num_agent[goalid] -= 1

            # if DEBUG: print("step", t)
            # if DEBUG: print("R_base", self.R_base[t, training_target])
            # if DEBUG: print("num_agent", num_agent[training_target])
            # if DEBUG: print("R_base[t, training_target]", self.R_base.shape, t, training_target)
            
            numera = 1.0 * ( self.R_base[t, :]  - num_agent[:]) # 分子
            reward1[i,:] = torch.where( (self.R_base[t,:] ==0) & (num_agent > 0), -torch.ones(num_agent.shape), torch.zeros(num_agent.shape) )
            reward1[i,:] = torch.where( (self.R_base[t,:] ==0) & (num_agent == 0), torch.zeros(num_agent.shape), reward1[i,:] )
            reward1[i,:] = torch.where( self.R_base[t,:] > 0 , numera / self.R_base[t, :], reward1[i,:] )
            reward1[i,:] = torch.where( reward1[i,:] < -1 , -torch.ones(num_agent.shape), reward1[i,:] )
            
            # reward1 = torch.max(-1, reward1)
            # if self.R_base[t, training_target] == 0:
            #     if num_agent[training_target] > 0:
            #         reward1[i,0] = -1
            #     else: # num_agent == 0:
            #         reward1[i,0] = 0
            # else:
            #     tmp_reward1 = 1.0 * ( self.R_base[t, training_target]  - num_agent[training_target]) / self.R_base[t, training_target] 
            #     if tmp_reward1 < -1:
            #         tmp_reward1 = -1
            #     reward1[i,0] = tmp_reward1

        if training_target is None:
            return reward1, self.R_base[t,:], num_agent # 全エージェント分
        else:
            return reward1[training_target], self.R_base[t, training_target], num_agent[training_target]

    def info2rewardArrive(self, infos, training_target=None, t=None):
        """
        収容人数から作成する報酬
        t: 時刻
        """
        if training_target is None:
            reward2 = torch.zeros((len(infos),len(self.group_agents)), dtype=torch.float64) # 全エージェント分
        else:
            reward2 = torch.zeros((len(infos),1), dtype=torch.float64) # 歩行人数から
        # reward2 = torch.zeros((len(infos),1)) # 収容人数から
        for i, info in enumerate(infos):
            goal_cnt = info["goal_cnt"]

            numera = 1.0 * ( self.A_base[t, :]  - goal_cnt[:]) # 分子
            reward2[i,:] = torch.where( (self.A_base[t,:] ==0) & (goal_cnt > 0), -torch.ones(goal_cnt.shape, dtype=torch.float64), torch.zeros(goal_cnt.shape, dtype=torch.float64) )
            reward2[i,:] = torch.where( (self.A_base[t,:] ==0) & (goal_cnt == 0), torch.zeros(goal_cnt.shape, dtype=torch.float64), reward2[i,:] )
            # print(self.A_base[t,:])
            # print(numera / self.A_base[t, :])
            # print(reward2)
            reward2[i,:] = torch.where( torch.tensor(self.A_base[t,:],dtype=torch.float64 ) > 0 , numera / self.A_base[t, :], reward2[i,:])
            reward2[i,:] = torch.where( reward2[i,:] < -1 , -torch.ones(goal_cnt.shape, dtype=torch.float64), reward2[i,:] )

            # if self.A_base[t, training_target] == 0:
            #     if goal_cnt[training_target] > 0:
            #         reward2[i,0] = -1
            #     else: # num_agent == 0:
            #         reward2[i,0] = 0
            # else:
            #     tmp_reward2 = 1.0 * ( self.A_base[t, training_target]  - goal_cnt[training_target]) / self.A_base[t, training_target] 
            #     if tmp_reward2 < -1:
            #         tmp_reward2 = -1
            #     reward2[i,0] = tmp_reward2
        # 符号を逆転させる->大きいほどよい
        if training_target is None:
            return -reward2, self.A_base[t,:], goal_cnt # 全エージェント分
        else:
            return -reward2[training_target], self.A_base[t, training_target], goal_cnt[training_target]
        # return -reward2, self.A_base[t, training_target], goal_cnt[training_target]

    def info2traveltime(self, infos):
        # エピソードの評価値を計算する
        ret = torch.zeros((len(infos),1))
        dict_travel_times = []
        for i, info in enumerate(infos):
            agentid, travel_time = info['goal_time']
            dict_travel_time = dict(zip(agentid, travel_time))
            dict_travel_times.append(dict_travel_time)
            if ("all_reached" in info) and (info["all_reached"] == True) :
                ret[i,0] = np.mean(travel_time)
            else:
                ret[i,0] = np.inf
        return ret, dict_travel_times

    def info2completetime(self, infos):
        # エピソードの評価値を計算する
        ret = torch.zeros((len(infos),1))
        dict_travel_times = []
        for i, info in enumerate(infos):
            agentid, travel_time = info['goal_time']
            dict_travel_time = dict(zip(agentid, travel_time))
            dict_travel_times.append(dict_travel_time)
            if ("all_reached" in info) and (info["all_reached"] == True) :
                ret[i,0] = np.max(travel_time)
            else:
                ret[i,0] = np.inf
        return ret, dict_travel_times

    def _get_reward_time(self): # mean travel time of people who reached goal
        agentid, travel_time = self._goal_time()
        # print(agentid, travel_time)
        if len(agentid) == 0:
            return 0
        reward = np.sum( self.travel_open[agentid] - travel_time ) / np.sum( self.travel_open[agentid] )
        if reward < 0:
            return max(reward, -1)
        return min(reward, 1)

    def _get_reward_total_time_once(self): # mean travel time of people who reached goal
        if self.travel_open is None:
            return 0
        if self.max_step > self.num_step + 1:
            return 0 # reward only last step
        agentid, travel_time = self._goal_time_all()
        # print(agentid, travel_time)
        # if len(agentid) == 0:
        #     return 0
        if len(agentid) != self.num_agents:
            return -1
        reward = np.sum( self.travel_open[agentid] - travel_time ) / np.sum( self.travel_open[agentid] )
        # reward = np.sum( self.T_open[agentid] - travel_time ) / np.sum( self.T_open[agentid] )
        if reward < 0:
            return max(reward, -1)
        return min(reward, 1)

    def _get_reward(self):
        # t_open
        tmp_state = np.sum( self.edge_state ) * self.interval / self.num_agents
        # print("T_open",self.T_open)
        if self.T_open is None:
            return tmp_state
        else:
            if self.T_open[self.num_step] == 0:
                if tmp_state == 0:
                    return 0
                else:
                    return -1
            reward = (self.T_open[self.num_step] - tmp_state) / (self.T_open[self.num_step])
            # print("tmp_reward", reward)
            if reward < 0:
                return max(reward, -1)
            return min(reward, 1)

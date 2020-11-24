#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from pedestrians import Crowd
import sys
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
        self.R_base = torch.zeros(len(T_open), len( self.crowds[0].first_dest.items() ))
        self.group_agents = [] # goalidごとに，agentidリストを保持
        for goalid, (nodeid, agentids) in enumerate( sorted( self.crowds[0].first_dest.items()) ):
            # print(goalid, (nodeid, agentids))
            print(goalid, len(agentids), np.mean( [dict_travel_time[agentid-1] for agentid in agentids] ) )
            self.group_agents.append(agentids)
        # print(dict_travel_time)

        for t, infos in enumerate(T_open):
            # num_pedestrian = np.sum( infos["edge_state"] )
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
        print(self.R_base)
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
            num_agent = torch.zeros(len(self.group_agents))
            agentid, travel_time = info['goal_time']
            # print(t, len(agentid), "人ゴール")
            # 当初目的地ごとに歩行者数を設定して
            for goalid, group_agent in enumerate( self.group_agents ):
                num_agent[goalid] = len(group_agent)
            # ゴールした人数を減算する
            for aid in agentid:
                goalid = self.edges[0].nodeid2goalid( self.crowds[0].pedestrians[aid-1].destination )
                num_agent[goalid] -= 1

            if DEBUG: print("step", t)
            if DEBUG: print("R_base", self.R_base[t, training_target])
            if DEBUG: print("num_agent", num_agent[training_target])

            if self.R_base[t, training_target] == 0:
                if num_agent[training_target] > 0:
                    reward[i,0] = -1
                    continue
                else: # num_agent == 0:
                    reward[i,0] = 0
                    continue
            else:
                tmp_reward = 1.0 * ( self.R_base[t, training_target]  - num_agent[training_target]) / self.R_base[t, training_target] 
                if tmp_reward < -1:
                    tmp_reward = -1
                reward[i,0] = tmp_reward

            # travel_time = info['travel_time']
            # agentid = info['agentid']
            # print(self.crowds)
            # print(self.crowds[0])
            # print(self.crowds[0].first_dest)
            # agentids = sorted( self.crowds[0].first_dest.items() )[training_target]
            # num_agent = len( [aid for aid in agentids if aid in agentid] ) 
            # reward[i,0] = 1.0 * ( num_agent - self.R_base[t, training_target] ) / self.R_base[t, training_target] 
            # for goalid, (nodeid, agentids) in enumerate( sorted( self.crowds[0].first_dest.items()) ):
            #     if goalid != training_target: # 学習対象の避難所が誘導した歩行者だけを考慮する
            #         continue
            #     num_agent = len( [aid for aid in agentids if aid in agentid] ) 
            #     reward[i,0] = 1.0 * ( num_agent - self.R_base[t, goalid] ) / self.R_base[t, goalid] 
            # tmp_travel_time = dict(zip(agentid, travel_time))
            #     reward[i,0] = np.mean( [tmp_travel_time[agentid] for agentid in agentids] )

        # reward = None
        return reward, self.R_base[t, training_target], num_agent[training_target]
        # return reward

    # def _get_reward_goal(self): # sum of people who reached goal
    #     G = np.sum( self._goal_cnt() )
    #     G_diff = G - self.prev_goal
    #     reward = G_diff / self.num_agents # - (1./ self.max_step)
    #     return reward, G

    # def _get_reward_goal_cum(self): # sum of people who reached goal
    #     G = np.sum( self._goal_cnt() )
    #     reward = G / self.num_agents # - (1./ self.max_step)
    #     return reward, G

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

    # def _get_reward_total_time_once_wo_norm(self): # mean travel time of people who reached goal
    #     if self.max_step > self.num_step + 1:
    #         return 0 # reward only last step
    #     agentid, travel_time = self._goal_time_all()
    #     # print(agentid, travel_time)
    #     # if len(agentid) == 0:
    #     #     return 0
    #     reward = np.sum( - travel_time ) / ( 1. * self.sim_time * self.num_agents)
    #     if reward < 0:
    #         return max(reward, -1)
    #     return min(reward, 1)

    # def _get_reward_speed(self, observation):
    #     # moving speedをRewardとする
    #     # 最大値を取るか，平均を取るかは要検討
    #     # 全歩行者数でRewardを決定
    #     v_mean = self.speed
    #     th     = 1.8 / (v_mean + 0.3)
    #     num_agent = observation[self.num_obsv * (self.obs_step-1):] # observationのサイズを変更したので要修正
    #     rho    = num_agent / (self.edges.dist * self.edges.width)
    #     # rho    = rho[self.num_edges * (self.obs_step-1):]
    #     rho_   = np.where(rho == 0, float('-inf'), rho)
    #     v_1    = np.where((0 <= rho) & (rho < th), v_mean, 0)
    #     v_2    = np.where((th <= rho_) & (rho_ < 6), 1.8 / rho_ - 0.3, 0)
    #     v      = (v_1 + v_2) * num_agent # weighted
    #     # reward = (v_mean - v) / v_mean
    #     if np.sum(num_agent) == 0:
    #         v = v_mean
    #     else:
    #         v = np.sum(v) / np.sum(num_agent)
    #     reward = ( v_mean - v ) / v_mean
    #     return -reward

    # def _get_reward_speed_w_V0(self, observation):
    #     # moving speedをRewardとする
    #     # 最大値を取るか，平均を取るかは要検討
    #     # 全歩行者数でRewardを決定
    #     v_mean = self.speed
    #     th     = 1.8 / (v_mean + 0.3)
    #     num_agent = observation[self.num_obsv * (self.obs_step-1):]
    #     rho    = num_agent / (self.edges.dist * self.edges.width)
    #     # rho    = rho[self.num_edges * (self.obs_step-1):]
    #     rho_   = np.where(rho == 0, float('-inf'), rho)
    #     v_1    = np.where((0 <= rho) & (rho < th), v_mean, 0)
    #     v_2    = np.where((th <= rho_) & (rho_ < 6), 1.8 / rho_ - 0.3, 0)
    #     v      = (v_1 + v_2) * num_agent # weighted
    #     # reward = (v_mean - v) / v_mean
    #     if np.sum(num_agent) == 0:
    #         v = v_mean
    #     else:
    #         v = np.sum(v) / np.sum(num_agent)
    #     # reward = ( v_mean - v ) / v_mean

    #     if self.V_open[self.num_step] == 0:
    #         if v == 0:
    #             return 0
    #         else:
    #             return 1
    #     reward = (self.V_open[self.num_step] - v) / (self.V_open[self.num_step])
    #     if reward < 0:
    #         return max(reward, -1)
    #     return min(reward, 1)

    # def _get_reward_wo_T0(self, observation):
    #     reward = ( self.num_agents - self._get_num_traveler(observation)) / (1. * self.num_agents)
    #     return reward

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

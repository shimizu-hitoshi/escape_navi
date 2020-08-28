#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import gym
import gym.spaces
from gym.utils import seeding
import numpy as np
import configparser
from cffi import FFI
sys.path.append('../')
from edges import Edge
# from controler import FixControler
import copy

DEBUG = True # False # True # False

class SimEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super().__init__()

    def step(self, action): # これを呼ぶときには，actionは決定されている
        # self.state = self._get_state() # 1stepにつき冒頭と末尾に状態取得？
        # print(action)
        # print(action.shape)
        # sys.exit()

        # if len(self.agents) == 1: # エージェント数が1だとスカラーになってエラー->暫定対処
        #     action = action.reshape(-1,1)

        # 他の避難所の行動を決定するコードを書く
        dict_actions = {} # key:ノードID, value:ノードID
        self.tmp_action_matrix = np.zeros(( len(self.agents),len(self.actions)), dtype=float)
        for i, node_id in enumerate( self.agents): # エージェントの行動
            # print("ここでtmp_action_matrixに代入する")
            # print("action.shape", action.shape)
            dict_actions[node_id] = self._get_action(action[i])
            self.tmp_action_matrix[i, action[i]] = 1.
        for shelter_id, node_id in enumerate( self.actions ): # エージェントではない避難所の行動
            if node_id in self.agents: # self.sidの代入と更新タイミングに注意
                continue
            _action = self.others[shelter_id].get_action(self.goal_state)
            dict_actions[node_id] = self._get_action(_action)
            # # for other in self.others:
            #     # if shelter_id == self.sid: # self.sidの代入と更新タイミングに注意
            #     if node_id == self.sid: # self.sidの代入と更新タイミングに注意
            #         # 自エージェントのactionだけ下で上書き
            #         dict_actions[node_id] = self._get_action(action)
            #     else:
            #         _action = self.others[shelter_id].get_action(self.goal_state)
            #         dict_actions[node_id] = self._get_action(_action)
        # _action = self._get_action(action)
        # self.call_traffic_regulation(_action, self.num_step)
        # print("dict_actions",dict_actions)
        self.call_traffic_regulation(dict_actions, self.num_step)
        self.call_iterate(self.cur_time + self.interval) # iterate
        self.cur_time += self.interval
        self.update_navi_state() # self.navi_stateを更新するだけ
        self.state = self._get_state()
        # self.state = self._get_observation(self.cur_time + self.interval) # iterate
        # observation = self.state2obsv( self.state, self.id ) 
        # observation = self.state 
        # reward = self._get_reward_time()
        # reward = self._get_reward()
        # reward = self._get_reward(self.edge_state)
        sum_pop = np.sum(self.edge_state) * self.interval / self.num_agents # 累積すると平均移動時間

        # if self.flg_reward == "goal":
        #     reward, G = self._get_reward_goal()
        #     self.prev_reward = copy.deepcopy(G)
        # elif self.flg_reward == "goal_cum":
        #     reward, _ = self._get_reward_goal_cum()
        #     # self.prev_reward = copy.deepcopy(G)
        # elif self.flg_reward == "edge_wo_T0":
        #     reward = self._get_reward_wo_T0(self.state)
        # elif self.flg_reward == "speed":
        #     reward = self._get_reward_speed(self.state)
        # elif self.flg_reward == "speed_w_V0":
        #     reward = self._get_reward_speed_w_V0(self.state)
        if DEBUG: print("in step(), flg_reward = ", self.flg_reward)
        if self.flg_reward == "time":
            reward = self._get_reward_time()
        elif self.flg_reward == "time_once":
            reward = self._get_reward_total_time_once()
        # elif self.flg_reward == "total_time_once_wo_norm":
        #     reward = self._get_reward_total_time_once_wo_norm()
        else: # self.flg_reward == "edge":
            reward = self._get_reward()

        # self.episode_reward += sum_pop
        self.episode_reward += reward
        # print("CURRENT", self.cur_time, action, sum_pop, self.T_open[self.num_step], reward, self.episode_reward)
        print("CURRENT", self.cur_time, action, sum_pop, reward, self.episode_reward)
        with open(self.resdir + "/current_log.txt", "a") as f:
            f.write("CURRENT {:} {:} {:} {:} {:}\n".format(self.cur_time, action, sum_pop, reward, self.episode_reward))
        self.num_step += 1
        done = self.max_step <= self.num_step
        # travel_time = self.mk_travel_open()
        info = {}
        if done:
            agentid, travel_time = self._goal_time_all() # 歩行者の移動速度リストを取得
            info = {
                    "episode": {
                        "r": self.episode_reward
                        },
                    "events": self.event_history,
                    "env_id":self.env_id,
                    "travel_time":travel_time,
                    "agentid":agentid
                    }
            # print("info",info)
        return self.state, reward, done, info # obseration, reward, done, info

    def reset(self):
        # config = configparser.ConfigParser()
        # config.read('config.ini')

        self.sim_time  = self.config.getint('SIMULATION', 'sim_time')
        self.interval  = self.config.getint('SIMULATION', 'interval')
        self.max_step  = int( np.ceil( self.sim_time / self.interval ))
        self.cur_time  = 0
        self.num_step  = 0
        self.state     = np.zeros(self.num_obsv * self.obs_step)

        self.episode_reward = 0
        self.event_history = []
        self.flag = True

        # for reward selection
        # self.prev_goal = 0

        self.reset_sim() # set up simulation

        return self.state

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None, env_id=None, datadirs=None, config=None, R_base=(None, None)):
        print(R_base)
        self.T_open, self.travel_open = R_base
        # print("T_open @ seed",self.T_open)
        # print("travel_open @ seed",self.travel_open)
        # training_targets = dict_target["training"]
        # fixed_agents = dict_target["fixed"] # その他を固定しよう
        # rule_agents = dict_target["rule"]
        # fixed_agents: モデルで行動，更新なし，の避難所
        # training_targets: 学習対象の避難所
        # rule_agents: ルールベースの避難所

        # from init (for config import)
        self.config = config
        # num_parallel   = config.getint('TRAINING',   'num_parallel')
        # tmp_id = len(training_targets) % num_parallel
        # tmp_id = seed % len(training_targets)
        tmp_id = env_id % len(datadirs)
        # if DEBUG: print(training_targets, tmp_id)
        self.env_id = env_id
        # self.sid = training_targets[tmp_id]
        # self.training_target = self.sid # 不要かも
        self.datadir = datadirs[tmp_id]
        # config = configparser.ConfigParser()
        # config.read('config.ini')
        # self.num_agents = config.getint('SIMULATION', 'num_agents')
        # self.num_edges  = config.getint('SIMULATION', 'num_edges')
        self.obs_step   = config.getint('TRAINING',   'obs_step')
        self.obs_degree   = config.getint('TRAINING',   'obs_degree')
        # self.datadir         = config.get('SIMULATION',    'datadir')
        self.tmp_resdir = config['TRAINING']['resdir']
        self.actions = np.loadtxt( config['SIMULATION']['actionfn'] , dtype=int )
        # self.agents = training_targets # = self.actions
        self.agents = copy.deepcopy(self.actions)
        if DEBUG: print(self.actions)
        # sys.exit()
        # self.dict_action = {}
        # for action in list( self.actions ):
        #     self.dict_action[]
        self.flg_reward = config['TRAINING']['flg_reward']

        self.init = True
        self.flag = True

        # self.edges = Edge(self.obs_degree) # degreeは不要になったはず．．．
        self.edges = Edge(self.datadir) # degreeは不要になったはず．．．
        # ->seed()の前に設定してしまいたい
        self.num_edges = self.edges.num_obsv_edge
        self.num_goals = self.edges.num_obsv_goal
        self.num_navi = len(self.actions) * len(self.actions) # 誘導の状態数は，ワンホットベクトルを想定
        self.navi_state = np.zeros(len(self.actions) * len(self.actions), dtype=float) # 入れ物だけ作っておく
        # self.num_navi = len(self.actions) * len(self.agents) # 誘導の状態数は，ワンホットベクトルを想定
        # self.navi_state = np.zeros(len(self.actions) * len(self.agents), dtype=float) # 入れ物だけ作っておく
        # self.num_obsv = self.num_edges + self.num_goals # １ステップ分の観測の数
        if DEBUG: print("self.navi_state.shape", self.navi_state.shape)
        self.num_obsv = self.num_edges + self.num_goals + self.num_navi # １ステップ分の観測の数

        self.action_space      = gym.spaces.Discrete(self.actions.shape[0])
        self.observation_space = gym.spaces.Box(
                low=0,
                high=100000,
                # high=self.num_agents,
                shape=np.zeros(self.num_obsv * self.obs_step).shape
                )
        assert self.action_space.n == self.actions.shape[0]
        assert self.observation_space.shape[0] == self.num_obsv * self.obs_step

        # self.state = None
        # self.state     = np.zeros(self.num_edges * self.obs_step)
        # self.cur_time  = 0
        # self.interval 
        # self.prev_goal = 0

        # copy from reset()
        self.sim_time  = self.config.getint('SIMULATION', 'sim_time')
        self.interval  = self.config.getint('SIMULATION', 'interval')
        self.max_step  = int( np.ceil( self.sim_time / self.interval ))
        self.cur_time  = 0
        self.num_step  = 0
        self.state     = np.zeros(self.num_obsv * self.obs_step)

        # original seed
        # self.np_random, seed = seeding.np_random(seed)
        # https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
        seeding.np_random(seed) 
        self.set_datadir(self.datadir)
        # print(self.datadir)
        self.set_resdir("%s/sim_result_%d"%(self.tmp_resdir, self.env_id))
        # ルールベースの避難所のエージェントを生成する
        # self.others = {}
        # for shelter_id, node_id in enumerate( self.actions ):
        #     # 自分のエージェントを作ってもいいけど，使わない
        #     controler = FixControler(shelter_id, self.edges.DistanceMatrix)
        #     self.others[shelter_id] = controler

        return [seed]

    def _get_action(self, action):
        return self.actions[action]

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

    def mk_travel_open(self): # travel time of people who reached goal
        stop_time = (self.max_step + 1) * self.interval
        start_time = 0
        print(start_time, stop_time)
        tmp = self.lib.goalAgentCnt(start_time, stop_time-1, -1) # all goal
        res = self.ffi.new("int[%d][3]" % tmp)
        l = self.lib.goalAgent(start_time, stop_time-1, tmp+1, res)
        # print(l)
        travel_time = np.array( [res[i][1] for i in range(l)] )
        # print(travel_time)
        return travel_time

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


    # def _get_travel_time(self, observation):
    #     # observation = observation[self.num_obsv * (self.obs_step-1):]
    #     observation = observation[self.num_edges * (self.obs_step-1):]
    #     return np.sum(observation) * self.interval / self.num_agents

    def _get_num_traveler(self, observation):
        observation = observation[self.num_obsv * (self.obs_step-1):]
        return np.sum(observation)

    def call_traffic_regulation(self, actions, t):
        """
        Take action, i.e., open/close gates
        """
        # pass
        events = self.get_events(actions, t)
        # for nid, action in actions.items():
        #     events.append(self.get_events(nid, action, t))
        # events = self.get_events(action, t)
        for e in events:
            self.lib.setBombDirect(e.encode('ascii'))
            if DEBUG: print(e)
            self.event_history.append(e)
        return events

    def _edge_cnt(self):
        ret   = np.zeros(len(self.edges.observed_edge))
        for e, idx in self.edges.observed_edge.items():
            fr, to     = e
            ret[idx]   = self.lib.cntOnEdge(fr-1, to-1)
        return ret

    def _goal_cnt(self): # ゴールした人の累積人数
        ret   = np.zeros(len(self.edges.observed_goal)) # 返値は，observed_goal次元
        stop_time = self.cur_time # + self.interval
        start_time = 0 # self.cur_time #stop_time - self.interval
        for node, idx in sorted( self.edges.observed_goal.items() ):
            tmp = self.lib.goalAgentCnt(start_time, stop_time-1, node-1)
            ret[idx]   = tmp
            # print(start_time, stop_time, node-1, idx, tmp)
        return ret

    def _goal_time(self):
        # 直近のstepにゴールした人の移動時間を求める
        stop_time = self.cur_time # + self.interval
        start_time = self.cur_time - self.interval
        tmp = self.lib.goalAgentCnt(start_time, stop_time-1, -1) # all goal
        res = self.ffi.new("int[%d][3]" % tmp)
        l = self.lib.goalAgent(start_time, stop_time-1, tmp+1, res)
        agentid = np.array( [res[i][0] for i in range(l)] )
        travel_time = np.array( [res[i][1] for i in range(l)] )
        return agentid, travel_time

    def _goal_time_all(self):
        # 現時点までにゴールした人の移動時間を求める
        stop_time = self.cur_time # + self.interval
        start_time = 0
        tmp = self.lib.goalAgentCnt(start_time, stop_time-1, -1) # all goal
        res = self.ffi.new("int[%d][3]" % tmp)
        l = self.lib.goalAgent(start_time, stop_time-1, tmp+1, res)
        agentid = np.array( [res[i][0] for i in range(l)] )
        travel_time = np.array( [res[i][1] for i in range(l)] )
        return agentid, travel_time

    def call_edge_cnt(self, stop_time=0):
        """
        Count the number of agents on the edge
        """
        self.lib.setStop(stop_time)
        self.lib.iterate()
        ret = self._edge_cnt()
        return ret

    def call_speed(self, stop_time=0):
        """
        Measure the mean of agents speed
        """
        # self.lib.setStop(stop_time)
        # self.lib.iterate()
        print("stop_time : ", stop_time)
        # self.state = self._get_observation(stop_time) # iterate
        self.call_iterate(stop_time)
        self.state = self._get_state()
        # observation = state2obsv( self.state, self.id ) 
        observation = self.state

        v_mean = self.speed
        th     = 1.8 / (v_mean + 0.3)
        num_agent = observation[self.num_edges * (self.obs_step-1):]
        rho    = num_agent / (self.edges.dist * self.edges.width)
        # rho    = rho[self.num_edges * (self.obs_step-1):]
        rho_   = np.where(rho == 0, float('-inf'), rho)
        v_1    = np.where((0 <= rho) & (rho < th), v_mean, 0)
        v_2    = np.where((th <= rho_) & (rho_ < 6), 1.8 / rho_ - 0.3, 0)
        v      = (v_1 + v_2) * num_agent # weighted
        # reward = (v_mean - v) / v_mean
        if np.sum(num_agent) == 0:
            v = v_mean
        else:
            v = np.sum(v) / np.sum(num_agent)
        return v

    # def call_goal_cnt(self, stop_time=0):
    #     """
    #     Count the number of agents on the goal
    #     """
    #     self.lib.setStop(stop_time)
    #     self.lib.iterate()
    #     # ret = np.zeros(len(self.edges.dict_edge))
    #     # for e, idx in self.edges.dict_edge.items():
    #     ret = _goal_cnt()
    #     return ret

    def call_iterate(self, stop_time=0):
        self.lib.setStop(stop_time)
        self.lib.iterate()

    def update_navi_state(self): # この関数，本当に必要？
        # self.navi_state = np.zeros(len(self.actions) * len(self.agents)) # 入れ物だけ作っておく
        # self.navi_state = self.tmp_action_matrix.reshape(1,-1)
        self.navi_state = self.tmp_action_matrix.flatten()
        # print( self.navi_state )
        # print( self.navi_state.shape )

    def _get_state(self):
        # １ステップ分ずらす
        obs     = self.state[self.num_obsv:] # 左端を捨てる
        # 避難所の状況を取得
        tmp_goal_state = copy.deepcopy( self._goal_cnt() )
        # print(tmp_goal_state)
        self.edge_state = copy.deepcopy( self._edge_cnt() ) # 何度も使いそうなので保存
        self.goal_state = self.edges.goal_capa - tmp_goal_state # 何度も使いそうなので保存
        # print(self.goal_state)
        # print(np.sum(tmp_goal_state), np.sum(self.edge_state))
        cur_obs = np.append(self.edge_state , self.goal_state )
        cur_obs = np.append(cur_obs , self.navi_state )
        return np.append(obs, cur_obs) # 右端に追加

    # def _get_observation(self, stop_time=0):
    #     self.call_iterate(stop_time)
    #     return self._get_state()
    #     # obs     = self.state[self.num_obsv:]
    #     # cur_obs = self.call_edge_cnt(stop_time) # iterate
    #     # return np.append(obs, cur_obs)

    def set_datadir(self, datadir):
        self.datadir = datadir
        agentfn    = os.path.dirname(os.path.abspath(__file__)) + "/../%s/agentlist.txt"%self.datadir
        self.speed = self.get_speed(agentfn)
        self.num_agents = self.get_num_agents(agentfn)

    def set_resdir(self, resdir):
        # print(resdir)
        os.makedirs(resdir, exist_ok=True)
        self.resdir = resdir

    def get_events(self, actions, t):
        # nid: 誘導元のノードID
        # action: 誘導先のノードID
        """
        たらい回さない：
        cmd_name="del_signage_table"
        tmp = [t,cmd_name,shelter, shelter]
        たらい回す：
        cmd_name = "signage"
        ratio = 1.0
        via_length = 0
        tmp = [t, cmd_name, shelter, shelter, target_node, target_node, ratio, via_length]
        # 末尾（1と0）は，分配率と経由地数
        """
        time = t * self.interval + 1
        out = []
        out.append( "%d clear_all_signage"%time )
        for nid, action in sorted( actions.items() ):
            if action == -1:
                pass
                # out = "%d del_signage_table %d %d"%(time, nid, nid)
            else:
                # detour = self.dict_action[action] 
                detour = action
                # if nid == action:
                # out = "%d del_signage_table %d %d"%(time, nid, nid)
                if nid != action:
                    out.append( "%d signage %d %d %d %d 1.0 0"%(time, nid, nid, detour, detour) )
        return out

    def call_open_event(self):
        # print(self.seed)
        # config = configparser.ConfigParser()
        # config.read('config.ini')
        # datadir = "mkUserlist/data/N80000r{:}i0".format(self.file_seed)
        datadir = self.datadir
        agentfn  = os.path.dirname(os.path.abspath(__file__)) + "/../%s/agentlist.txt"%self.datadir
        graphfn  = os.path.dirname(os.path.abspath(__file__)) + "/../%s/graph.twd"%self.datadir
        goalfn   = os.path.dirname(os.path.abspath(__file__)) + "/../%s/goallist.txt"%self.datadir
        sim_time = self.config.get('SIMULATION', 'sim_time')
        resdir = self.resdir
        # resdir = "result/%s"%self.datadir
        print(resdir)
        os.makedirs(resdir, exist_ok=True)
        argv = [sys.argv[0]]
        argv.extend([
            agentfn,
            graphfn,
            goalfn,
            "-o",
            # "bin/result%s"%self.seed,
            resdir,
            "-l",
            "9999999",
            # "300", # ログ周期
            # "10", # ログ周期
            "-e",
            sim_time,
            # "-S"
            ])
        # print(argv)
        tmp = []
        for a in argv:
            # tmp.append(self.ffi.new("char []", a))
            # tmp.append(self.ffi.new("char []", a.encode('UTF-8')))
            tmp.append(self.ffi.new("char []", a.encode('ascii')))
        argv = self.ffi.new("char *[]", tmp)
        # call simulator
        self.lib.init(len(argv), argv)
        # _action = self._get_action(0)
        # print(_action)
        # self.call_traffic_regulation({}, 0)
        # self.T_open = None
        # call_open_event全体をコメントアウトすると動作しないので，ここまでだけ実行（暫定）
        return 
        # elif "speed_w_V0" == self.flg_reward:
        #     print("self.call_speed")
        #     self.V_open = [self.call_speed((i + 1) * self.interval) for i in range(self.max_step)]
        #     print(self.V_open, np.sum(self.V_open))
        # elif "time" == self.flg_reward or "time_once" == self.flg_reward:
        #     print("travel time")
        #     self.call_iterate( (self.max_step + 1) * self.interval)
        #     self.travel_open = self.mk_travel_open() # ここにベースラインを保存する
        #     print(self.travel_open, len(self.travel_open), np.mean(self.travel_open))
        # else:
        #     pass

    def str_cdef(self):
        ret = """
        void init(int argc, char** argv);
        int   setStop(int t);
        void  iterate();
        int   cntDest(int node, double radius);
        int   cntSrc(int node, double radius);
        void  setBomb( char *fn);
        int   cntOnEdge(int fr, int to);
        void  setBombDirect(char *text);
        void  restart();
        void  save_ulog(char *fn);
        void  init_restart(int argc, char** argv);
        int   goalAgentCnt(int stime, int etime, int cnt);
        int   goalAgent(int stime, int etime,int n,  int result[][3]);
        """
        return ret

    def reset_sim(self):
        print("reset simulator configure")
        if self.init:
            libsimfn = os.path.dirname(os.path.abspath(__file__)) + "/../bin/libsim.so"
            self.ffi = FFI()
            self.lib = self.ffi.dlopen(libsimfn)
            self.ffi.cdef(self.str_cdef())
            self.call_open_event()
            self.num_step  = 0 # call_open_event中にstep進めてしまってるので
            self.cur_time  = 0 # 同上
            print("INIT")
            self.lib.restart()
            # save initial state
            self.init = False
        else:
            # load inital state
            self.lib.restart()

    def get_speed(self, agentfn):
        with open(agentfn) as f:
            lines = f.readlines()
        return sum([float(l.split('\t')[2]) for l in lines[1:]]) / float(lines[0].split(' ')[0])

    def get_num_agents(self, agentfn):
        with open(agentfn) as f:
            lines = f.readlines()
        return int(lines[0].split(' ')[0])

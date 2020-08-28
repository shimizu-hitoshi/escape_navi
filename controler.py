# import torch.nn as nn
# import torch.nn.functional as F
# def init(module, gain):
#     nn.init.orthogonal_(module.weight.data, gain=gain)
#     nn.init.constant_(module.bias.data, 0)
#     return module
import torch

DEBUG = True # False

class FixControler():
    def __init__(self, shelter_id, edges):
        # super(FixControler, self).__init__()
        self.sid = shelter_id # 道路網ノードIDではなく，避難所のみに0から採番
        self.num_edges = edges.num_obsv_edge
        self.num_goals = edges.num_obsv_goal
        # num_naviがないので，self.num_goal*self.num_goalで代用
        self.num_obsv = self.num_edges + self.num_goals + (self.num_goals*self.num_goals)
        DistanceMatrix = edges.DistanceMatrix
        self.cands = []
        # self.DistanceMatrix = DistanceMatrix
        num_shelter = DistanceMatrix.shape[0]
        tmp = {}
        for m in range(num_shelter):
            tmp[m] = DistanceMatrix[self.sid,m]
        self.cands = sorted( tmp.items(), key=lambda x:x[1])
        self.cands = [i[0] for i in self.cands]
        if DEBUG: print("shelter_id", shelter_id, self.cands)

    def get_action(self, x):
        # xは，避難所の残容量ベクトル
        # print("get_action",x.shape)
        # ret = torch.zeros([x.shape[0], 1])
        # for j in range(x.shape[0]):
        for i in self.cands:
            if x[i] > 0:
                return i
        return self.sid
        # ret[j,0] = j # 残容量のある避難所が見つからなければ自分
        # for i in self.cands:
        #     # if i == 18: # debug
        #     #     print(x)
        #     if x[i] > 0:
        #         return i 
        # return -1 # 空き容量のある避難所が見つからなければ-1
        # return ret

    def act_greedy(self,obs):
        # print("act_greedy", self.num_edges)
        if DEBUG: print("act_greedy", obs.shape)
        current_obs     = obs[:,-self.num_obsv:] # 右端が現在時刻
        # x = obs[:,self.num_edges:(self.num_edges+self.num_goals)] # 状態の冒頭に道路上人数，次に残容量がある想定
        x = current_obs[:,self.num_edges:(self.num_edges+self.num_goals)] # 現在の状態の冒頭に道路上人数，次に残容量がある想定
        if DEBUG: print("sid", self.sid)
        if DEBUG: print("act_greedy", x.shape)
        if DEBUG: print("x=", x)
        # print("act_greedy", x.shape, x)
        ret = torch.zeros([x.shape[0], 1])
        for j in range(x.shape[0]):
            ret[j,0]  = self.get_action(x[j,:])
        # print(ret)
        if DEBUG: print("action=", ret)
        return ret

    def act(self,obs):
        return self.act_greedy(obs)

if __name__=="__main__":
    import numpy as np

    d = np.random.rand(10,10)
    d = np.zeros((10,10))
    c = FixControler(1,d)
    x = np.random.rand(10)
    print(x)
    a = c.get_action(-x)
    print(a)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

class RewardMaker(object):
    def __init__(self):
        pass

    def set_R_base(self, R_base):
        T_open, dict_travel_time = R_base
        self.T_open = T_open
        self.dict_travel_time = dict_travel_time

    def info2reward(self, infos, training_target):
        print(infos)
        reward = torch.zeros((len(infos),1))
        for i, info in enumerate(infos):
            if "travel_time" not in info:
                continue
            travel_time = info['travel_time']
            agentid = info['agentid']
            tmp_travel_time = dict(zip(agentid, travel_time))

        # reward = None
        return reward


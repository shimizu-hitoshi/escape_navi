import torch
import torch.nn as nn
import torch.nn.functional as F
from controler import FixControler

def init(module, gain):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module

class ActorCritic(nn.Module):
    def __init__(self, n_in, n_out):
        super(ActorCritic, self).__init__()

        def init_(module): return init(module, gain=nn.init.calculate_gain('relu'))
        
        self.n_out = n_out
        self.better_agents = [] # ルールベースを超えたsidのリスト
        mid_io = 128
        self.linear1 = nn.Linear(n_in, mid_io)
        self.linear2 = nn.Linear(mid_io, mid_io)
        self.linear3 = nn.Linear(mid_io, mid_io)

        # self.actor   = nn.Linear(mid_io, n_out) # この行たぶん不要
        actors = [nn.Linear(mid_io, n_out) for _ in range(n_out)]
        self.actors   = nn.ModuleList(actors)
        # nn.init.normal_(self.actor.weight, 0.0, 1.0)
        self.critic  = nn.Linear(mid_io, 1)
        # nn.init.normal_(self.critic.weight, 0.0, 1.0)

    def set_FixControler(self, edges):
        # 20201116: model内部にFixControlerを保持するように改造
        self.dict_FixControler = {}
        # dict_FixControlerに，ルールベースのエージェントを配置しておく
        for sid in range(self.n_out):
            self.dict_FixControler[sid] = FixControler(sid, edges)
        self.eps = 1.0 # ルールベースを選択する確率
        self.loop_i = 0 # 学習した回数を自分で覚える
        self.learn_time = 5000 # 500回学習したらルールベースを使わなくなる

    def forward(self, x, sid):
        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(h1))
        h3 = F.relu(self.linear3(h2))

        critic_output = self.critic(h3)
        actor_output  = self.actors[sid](h3)
        # actor_output  = self.actor(h3)
        return critic_output, actor_output

    def set_edges(self, edges):
        self.num_edges = edges.num_obsv_edge
        self.num_goals = edges.num_obsv_goal

    def legal_actions(self, obs):
        x = obs[:,self.num_edges:(self.num_edges+self.num_goals)] # 状態の冒頭に道路上人数，次に残容量がある想定
        ret = torch.where( x > 0 )
        if ret[0].shape[0] == 0: # どの避難所も残容量がなくなったら，全避難所を選択対象にする
            ret = torch.where( x == 0 )
        # print("legal_action: ret",ret)
        return ret

    def act(self, x, sid, flg_greedy=False, flg_legal=True):
        value, actor_output = self(x, sid)
        if flg_legal: # 空いている避難所のみを誘導先候補にする
            ret = torch.zeros(x.shape[0],1)
            legal_actions = self.legal_actions(x)
            for i in range(x.shape[0]):
                if self.eps > torch.rand(1): # 一定確率でルールベースを選択
                    # print("fix")
                    tmp_action = self.dict_FixControler[sid].act_greedy(x)
                    #     tmp_action = self.dict_FixControler[i].act_greedy(x)
                    ret[i,0] = tmp_action[i,:]
                    # ret[i,0] = tmp_action.squeeze()
                else:
                    # print("not fix")
                    idxs = legal_actions[1][legal_actions[0]==i]
                    if flg_greedy:
                        action_probs = F.softmax(actor_output[i,idxs], dim=0).detach()
                        # print(action_probs.shape)
                        # print(action_probs.data.max(0))
                        tmp_action = action_probs.data.max(0)[1].view(-1, 1)
                    else: # 全避難所を誘導先候補にする
                        action_probs = F.softmax(actor_output[i,idxs], dim=0)
                        tmp_action = action_probs.multinomial(num_samples=1)
                    action = idxs[tmp_action]
                    ret[i,0] = action
            # print(ret.shape)
            return ret
        else:
            if flg_greedy:
                action_probs = F.softmax(actor_output, dim=1).detach()
                action       = action_probs.data.max(1)[1].view(-1, 1)
            else:
                action_probs = F.softmax(actor_output, dim=1)
                action       = action_probs.multinomial(num_samples=1)
            # print(action.shape)
            return action
 
    def act_greedy(self, x, sid):
        return self.act(x, sid, flg_greedy=True)

        # value, actor_output = self(x)
        # legal_actions = self.legal_actions(x)
        # action_probs = F.softmax(actor_output[legal_actions], dim=1).detach()
        # # print(action_probs)
        # # action       = action_probs.data.max(1)[1].view(1, 1)
        # tmp_action       = action_probs.data.max(1)[1].view(-1, 1)
        # action = legal_actions[tmp_action]
        # # action       = action_probs.data.max(1)[1].view(-1, 1)
        # return action

    def get_value(self, x, sid):
        # return state-value
        value, actor_output = self(x, sid)
        return value

    def evaluate_actions(self, x, actions, sid, flg_legal=True):

        value, actor_output = self(x, sid)
        log_probs = F.log_softmax(actor_output, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        # probs   = F.softmax(actor_output, dim=1)
        probs = torch.zeros(actor_output.shape)

        if flg_legal: # 空いている避難所のみを誘導先候補にする
            legal_actions = self.legal_actions(x)
            for i in range(x.shape[0]):# i:並列SimのID
                idxs = legal_actions[1][legal_actions[0]==i]
                action_probs = F.softmax(actor_output[i,idxs], dim=0)
                probs[i,idxs] = action_probs
        else:
            pass

        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy

    def set_better_agents(self, better_agents):
        self.better_agents = better_agents

    def act_all(self, x, training_target):
        action = torch.zeros(x.shape[0], self.n_out).long() # .to(self.device) # 各観測に対する，各エージェントの行動
        for i in range( self.n_out ):
            if i == training_target:
                tmp_action = self.act(x, i)
                # target_action = copy.deepcopy(tmp_action)
            # elif i in self.better_agents:
            else:
                tmp_action = self.act_greedy(x, i)
            # elif i in self.better_agents:
            #     # print(actor_critic.__class__.__name__)
            #     tmp_action = self.act_greedy(x, i) # ここでアクション決めて
            # else:
            #     tmp_action = self.dict_FixControler[i].act_greedy(x)
            action[:,i] = tmp_action.squeeze()
        return action

    def test_act_all(self, x, test_list, fix_list): # 評価用=softmaxしない
        action = torch.zeros(x.shape[0], self.n_out).long() # .to(self.device) # 各観測に対する，各エージェントの行動
        for i in range( self.n_out ):
            if ((i in self.better_agents) or (i in test_list)) and (i not in fix_list):
                tmp_action = self.act_greedy(x, i) # ここでアクション決めて
            else:
                tmp_action = self.dict_FixControler[i].act_greedy(x)
            action[:,i] = tmp_action.squeeze()
        return action
    
    def update_eps(self):
        tmp_init = 0.9 # loop_i=0のときのepsの値
        self.eps = max( tmp_init - (tmp_init * self.loop_i) / (1.0 * self.learn_time) , 0.0)

class ActorN_CriticN(ActorCritic):
    def __init__(self, n_in, n_out):
        # super(ActorN_CriticN, self).__init__()
        super().__init__(n_in, n_out)

        def init_(module): return init(module, gain=nn.init.calculate_gain('relu'))
        
        self.n_out = n_out
        self.better_agents = [] # ルールベースを超えたsidのリスト
        mid_io = 128
        self.linear1 = nn.Linear(n_in, mid_io)
        self.linear2 = nn.Linear(mid_io, mid_io)
        self.linear3 = nn.Linear(mid_io, mid_io)

        actors = [nn.Linear(mid_io, n_out) for _ in range(n_out)]
        self.actors   = nn.ModuleList(actors)
        critics = [nn.Linear(mid_io, 1) for _ in range(n_out)]
        self.critics  = nn.ModuleList(critics)

    def forward(self, x, sid):
        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(h1))
        h3 = F.relu(self.linear3(h2))

        critic_output = self.critics[sid](h3)
        actor_output  = self.actors[sid](h3)
        # actor_output  = self.actor(h3)
        return critic_output, actor_output

class ActorN_CriticN_share2(ActorCritic):
    def __init__(self, n_in, n_out):
        # super(ActorN_CriticN, self).__init__()
        super().__init__(n_in, n_out)

        def init_(module): return init(module, gain=nn.init.calculate_gain('relu'))
        
        self.n_out = n_out
        self.better_agents = [] # ルールベースを超えたsidのリスト
        mid_io = 128
        self.linear1 = nn.Linear(n_in, mid_io)
        self.linear2 = nn.Linear(mid_io, mid_io)
        layter3s = [nn.Linear(mid_io, mid_io) for _ in range(n_out)]
        self.linear3 = nn.ModuleList(layter3s)
        # self.linear3 = nn.Linear(mid_io, mid_io)

        actors = [nn.Linear(mid_io, n_out) for _ in range(n_out)]
        self.actors   = nn.ModuleList(actors)
        critics = [nn.Linear(mid_io, 1) for _ in range(n_out)]
        self.critics  = nn.ModuleList(critics)

    def forward(self, x, sid):
        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(h1))
        h3 = F.relu(self.linear3[sid](h2))

        critic_output = self.critics[sid](h3)
        actor_output  = self.actors[sid](h3)
        # actor_output  = self.actor(h3)
        return critic_output, actor_output


class ActorN_CriticN_share1(ActorCritic):
    def __init__(self, n_in, n_out):
        # super(ActorN_CriticN, self).__init__()
        super().__init__(n_in, n_out)

        def init_(module): return init(module, gain=nn.init.calculate_gain('relu'))
        
        self.n_out = n_out
        self.better_agents = [] # ルールベースを超えたsidのリスト
        mid_io = 128
        self.linear1 = nn.Linear(n_in, mid_io)
        layter2s = [nn.Linear(mid_io, mid_io) for _ in range(n_out)]
        self.linear2 = nn.ModuleList(layter2s)
        # self.linear2 = nn.Linear(mid_io, mid_io)
        layter3s = [nn.Linear(mid_io, mid_io) for _ in range(n_out)]
        self.linear3 = nn.ModuleList(layter3s)
        # self.linear3 = nn.Linear(mid_io, mid_io)

        actors = [nn.Linear(mid_io, n_out) for _ in range(n_out)]
        self.actors   = nn.ModuleList(actors)
        critics = [nn.Linear(mid_io, 1) for _ in range(n_out)]
        self.critics  = nn.ModuleList(critics)

    def forward(self, x, sid):
        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2[sid](h1))
        h3 = F.relu(self.linear3[sid](h2))

        critic_output = self.critics[sid](h3)
        actor_output  = self.actors[sid](h3)
        # actor_output  = self.actor(h3)
        return critic_output, actor_output


class ActorN_CriticN_share0(ActorCritic):
    def __init__(self, n_in, n_out):
        # super(ActorN_CriticN, self).__init__()
        super().__init__(n_in, n_out)

        def init_(module): return init(module, gain=nn.init.calculate_gain('relu'))
        
        self.n_out = n_out
        self.better_agents = [] # ルールベースを超えたsidのリスト
        mid_io = 128

        layter1s = [nn.Linear(n_in, mid_io) for _ in range(n_out)]
        self.linear1 = nn.ModuleList(layter1s)
        # self.linear1 = nn.Linear(n_in, mid_io)
        layter2s = [nn.Linear(mid_io, mid_io) for _ in range(n_out)]
        self.linear2 = nn.ModuleList(layter2s)
        # self.linear2 = nn.Linear(mid_io, mid_io)
        layter3s = [nn.Linear(mid_io, mid_io) for _ in range(n_out)]
        self.linear3 = nn.ModuleList(layter3s)
        # self.linear3 = nn.Linear(mid_io, mid_io)

        actors = [nn.Linear(mid_io, n_out) for _ in range(n_out)]
        self.actors   = nn.ModuleList(actors)
        critics = [nn.Linear(mid_io, 1) for _ in range(n_out)]
        self.critics  = nn.ModuleList(critics)

    def forward(self, x, sid):
        h1 = F.relu(self.linear1[sid](x))
        h2 = F.relu(self.linear2[sid](h1))
        h3 = F.relu(self.linear3[sid](h2))

        critic_output = self.critics[sid](h3)
        actor_output  = self.actors[sid](h3)
        # actor_output  = self.actor(h3)
        return critic_output, actor_output

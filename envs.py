import gym
import torch
import copy
from model import ActorCritic
from brain import Brain
from storage import RolloutStorage
from controler import FixControler
from mp.envs import make_vec_envs
import json
import numpy as np
import configparser
from torch.autograd import Variable
import shutil
import os, sys, glob
from edges import Edge

DEBUG = False # True # False # True # False

class Curriculum:
    def run(self, args):
        config = configparser.ConfigParser()
        config.read(args.configfn)
        test_env = Environment(args, "test")
        training_targets = list( np.loadtxt( config['TRAINING']['training_target'] , dtype=int ) )
        shelters = np.loadtxt( config['SIMULATION']['actionfn'] , dtype=int )
        edgedir = config['SIMULATION']['edgedir'] # datadir の代用
        edges = Edge(edgedir) # 暫定

        print(config['CURRICULUM'])
        outputfn = config['CURRICULUM']['outputfn'] # model file name
        resdir = config['CURRICULUM']['resdir']
        if not os.path.exists(resdir):
            os.makedirs(resdir)
        print(resdir)

        # 設定を保存
        shutil.copy2(args.configfn, resdir)
        with open(resdir + "/args.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)

        dict_best_model = {}

        dict_model = {}
        if args.checkpoint:
            # モデルを読み込む処理
            ifns = glob.glob(args.inputfn + "_*")
            for ifn in ifns:
                print("loading: ",ifn)
                node_id = int( ifn.split("_")[-1] )
                actor_critic = load_model(test_env.n_in, test_env.n_out, ifn).to(test_env.device)
                actor_critic.set_edges(edges)
                dict_model[node_id] = actor_critic

        # 最初は，ルールベースのエージェントを配置しておく
        for sid, shelter in enumerate(shelters):
            controler = FixControler(sid, edges)
            if shelter in dict_model: # モデルを読み込んだnodeはスキップ
                continue
            dict_model[shelter] = controler

        # sys.exit()
        best_score, R_base = test_env.test(dict_model) # ルールベースの評価値を取得
        print("初回のスコア", best_score, R_base)

        if args.test: # testモードなら，以下の学習はしない
            sys.exit()

        dict_best_model = copy.deepcopy(dict_model)
        # tmp_fixed = copy.deepcopy(dict_target["training"])
        i_Curriculum = 0 # カリキュラムのループカウンタ
        while True:
            # 突然エラー出たので，毎回インスタンス生成するように修正
            train_env = Environment(args, "train", R_base, i_Curriculum)
            flg_update = False
            for training_target in training_targets:
                # dict_target["training"] = [training_target]
                # dict_target["fixed"] = tmp_fixed
                # dict_target["fixed"].remove(training_target)
                dict_model = copy.deepcopy(dict_best_model)
                # targetがまだデフォルト制御なら，新規にエージェントを生成する
                if dict_best_model[training_target].__class__.__name__ == "FixControler":
                    dict_model[training_target] = ActorCritic(train_env.n_in, train_env.n_out)
                    dict_model[training_target].set_edges(edges)

                dict_model = train_env.train(dict_model, config, training_target)
                tmp_score, _ = test_env.test(dict_model)
                if tmp_score < best_score: # scoreは移動時間なので小さいほどよい
                    best_score = copy.deepcopy(tmp_score)
                    for node_id, model in dict_model.items():
                        dict_best_model[node_id] = copy.deepcopy(model)
                    flg_update = True
                else: # 性能を更新できなかったら，戻す
                    dict_best_model[training_target] = dict_best_model[training_target]
            if not flg_update: # 1個もtargetが更新されなかったら終了
                break
        # モデルを保存して終了

        if args.save:
            # save_model(actor_critic, resdir + '/' + outputfn)
            for node_id, model in dict_best_model.items():
                if model.__class__.__name__ == "FixControler":
                    print("node", node_id, " is FixControler")
                else:
                    print(resdir + '/' + outputfn + "_%s"%node_id +"をセーブする")
                    save_model(model, resdir + '/' + outputfn + "_%s"%node_id )
        print("ここでCurriculum終了")


class Environment:
    def __init__(self, args, flg_test=False, R_base=None):
        config = configparser.ConfigParser()
        config.read(args.configfn)
        # config.read('config.ini')
        self.sim_time  = config.getint('SIMULATION', 'sim_time')
        self.interval  = config.getint('SIMULATION', 'interval')
        self.max_step  = int( np.ceil( self.sim_time / self.interval ))

        # NUM_PROCESSES     = config.getint('TRAINING', 'num_processes')
        if flg_test=="test":
            self.NUM_PARALLEL = 1
        else:
            self.NUM_PARALLEL     = config.getint('TRAINING', 'num_parallel')

        self.NUM_ADVANCED_STEP = config.getint('TRAINING', 'num_advanced_step')
        self.NUM_EPISODES      = config.getint('TRAINING', 'num_episodes')
        # outputfn = config['TRAINING']['outputfn'] # model file name
        self.gamma = float( config['TRAINING']['gamma'] )
        self.datadirs = []
        with open(config['SIMULATION']['datadirlistfn']) as fp:
            for line in fp:
                datadir = line.strip()
                self.datadirs.append(datadir)
        # training_targets = list( np.loadtxt( config['TRAINING']['training_target'] , dtype=int ) )
        # これを引数で指定
        # self.NUM_PROCESSES = NUM_PARALLEL * NUM_AGENTS

        print(config['TRAINING'])
        self.resdir = config['TRAINING']['resdir']
        if not os.path.exists(self.resdir):
            os.makedirs(self.resdir)
        # print(resdir)
        # shutil.copy2(args.configfn, self.resdir)
        # with open(self.resdir + "/args.txt", "w") as f:
        #     json.dump(args.__dict__, f, indent=2)

        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        self.envs = make_vec_envs(args.env_name, args.seed, self.NUM_PARALLEL, self.device, self.datadirs, config, R_base)
        self.n_in  = self.envs.observation_space.shape[0]
        self.n_out = self.envs.action_space.n
        self.obs_shape       = self.n_in

    # def set_R_base(self, R_base):
    #     self.envs.set_R_base(R_base)

    def train(self, dict_model, config, training_target):
        self.NUM_AGENTS = len(dict_model)
        # print("train", dict_model)
        # actor_critics = []
        # local_brains = []
        # rollouts = []
        actor_critic = dict_model[training_target]
        global_brain = Brain(actor_critic, config)
        rollout = RolloutStorage(self.NUM_ADVANCED_STEP, self.NUM_PARALLEL, self.obs_shape, self.device)

        # for i, (k,v) in enumerate( dict_model.items() ):
        #     actor_critic = dict_model[k]
        #     if actor_critic.__class__.__name__ == "FixControler":
        #         continue
        #     actor_critics.append(v)
        #     # global_brain = Brain(actor_critic, config)
        #     local_brain = Brain(actor_critic, config)
        #     local_brains.append(local_brain)

        #     # rollouts        = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape, device)
        #     rollout = RolloutStorage(self.NUM_ADVANCED_STEP, self.NUM_PARALLEL, self.obs_shape, self.device)
        #     rollouts.append(rollout)

        current_obs     = torch.zeros(self.NUM_PARALLEL, self.obs_shape).to(self.device)
        episode_rewards = torch.zeros([self.NUM_PARALLEL, 1])
        final_rewards   = torch.zeros([self.NUM_PARALLEL, 1])

        episode         = np.zeros(self.NUM_PARALLEL)

        obs = self.envs.reset()
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()
        current_obs = obs

        # envs.set_t_open(T_open)
        rollout.observations[0].copy_(current_obs)

        while True:
            for step in range(self.NUM_ADVANCED_STEP):
                with torch.no_grad():
                    # action = actor_critic.act(rollouts.observations[step]) # ここでアクション決めて
                    action = torch.zeros(self.NUM_PARALLEL, self.NUM_AGENTS).long().to(self.device) # 各観測に対する，各エージェントの行動
                    if DEBUG: print("actionサイズ",self.NUM_PARALLEL, self.NUM_AGENTS)
                    for i, (k,v) in enumerate( dict_model.items() ):
                        if k == training_target:
                            tmp_action = v.act(current_obs)
                            target_action = copy.deepcopy(tmp_action)
                        else:
                            tmp_action = v.act_greedy(current_obs)
                        action[:,i] = tmp_action.squeeze()
                    # for i, (actor_critic, training_target) in enumerate( zip(actor_critics, training_targets) ):
                    #     if training_target in fixed_targets:
                    #         tmp_action = actor_critic.act_greedy(rollouts[i].observations[step]) # ここでアクション決めて
                    #     else:
                    #         tmp_action = actor_critic.act(rollouts[i].observations[step]) # ここでアクション決めて
                    #    action[:,i] = tmp_action.squeeze()
                        # if NUM_AGENTS > 1:
                        #     action[:,i] = tmp_action.squeeze()
                        # else:
                        #     action = tmp_action
                if DEBUG: print("step前のここ？",action.shape)
                obs, reward, done, infos = self.envs.step(action) # これで時間を進める
                episode_rewards += reward

                # if done then clean the history of observation
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                if DEBUG: print("done.shape",done.shape)
                if DEBUG: print("masks.shape",masks.shape)
                if DEBUG: print("obs.shape",obs.shape)
                with open(self.resdir + "/episode_reward.txt", "a") as f:
                    for i, info in enumerate(infos):
                        if 'episode' in info:
                            f.write("{:}\t{:}\t{:}\t{:}\n".format(training_target,episode[i], info['env_id'], info['episode']['r']))
                            print(training_target, episode[i], info['env_id'], info['episode']['r'])
                            episode[i] += 1

                final_rewards *= masks
                final_rewards += (1-masks) * episode_rewards

                episode_rewards *= masks
                current_obs     *= masks

                current_obs = obs # ここで観測を更新している

                rollout.insert(current_obs, target_action.data, reward, masks, self.NUM_ADVANCED_STEP)
                with open(self.resdir + "/reward_log.txt", "a") as f:
                    f.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format(training_target, episode.mean(), step, reward.max().numpy(), reward.min().numpy(), reward.mean().numpy(), episode_rewards.max().numpy(), episode_rewards.min().numpy(), episode_rewards.mean().numpy()))
                    print(training_target, episode.mean(), step, reward.mean().numpy(), episode_rewards.mean().numpy())

            with torch.no_grad():
                next_value = actor_critic.get_value(rollout.observations[-1]).detach()

            rollout.compute_returns(next_value, self.gamma)
            value_loss, action_loss, total_loss, entropy = global_brain.update(rollout)

            with open(self.resdir + "/loss_log.txt", "a") as f:
                f.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format(training_target, episode.mean(), value_loss, action_loss, entropy, total_loss))
                print("value_loss {:}\taction_loss {:}\tentropy {:}\ttotal_loss {:}".format(value_loss, action_loss, entropy, total_loss))

            rollout.after_update()
            
            if int(episode.mean())+1 > self.NUM_EPISODES:
                # print("ループ抜ける")
                break
        # ここでベストなモデルを保存していた（備忘）
        print("%s番目のエージェントのtrain終了"%training_target)
        dict_model[training_target] = actor_critic # {}
        return dict_model

    def test(self, dict_model): # 1並列を想定する
        self.NUM_AGENTS = len(dict_model)
        # config = configparser.ConfigParser()
        # config.read(args.configfn)
        # sim_time  = config.getint('SIMULATION', 'sim_time')
        # interval  = config.getint('SIMULATION', 'interval')
        # max_step  = int( np.ceil( sim_time / interval ))

        # datadirs = []
        # with open(config['SIMULATION']['datadirlistfn']) as fp:
        #     for line in fp:
        #         datadir = line.strip()
        #         datadirs.append(datadir)
        # print(dict_target)
        # training_targets = dict_target["training"]
        # # training_targets = list( np.loadtxt( config['TRAINING']['training_target'] , dtype=int ) )
        # NUM_AGENTS = len(training_targets)
        # # NUM_PROCESSES = NUM_PARALLEL * NUM_AGENTS

        # print(config['TEST'])
        # resdir = config['TEST']['resdir']
        # if not os.path.exists(resdir):
        #     os.makedirs(resdir)
        # print(resdir)
        # shutil.copy2(args.configfn, resdir)
        # with open(resdir + "/args.txt", "w") as f:
        #     json.dump(args.__dict__, f, indent=2)

        # device = torch.device("cuda:0" if args.cuda else "cpu")
    
        # envs = make_vec_envs(args.env_name, args.seed, 1, device, datadirs, dict_target, config)
        # n_in  = envs.observation_space.shape[0]
        # n_out = envs.action_space.n
        # obs_shape       = n_in
        NUM_PARALLEL = 1
        actor_critics = []
        # for i, training_target in enumerate( training_targets ):
        for i, actor_critic in sorted( dict_model.items() ):
            actor_critics.append(actor_critic)

        current_obs     = torch.zeros(self.NUM_PARALLEL, self.obs_shape).to(self.device)
        episode_rewards = torch.zeros([self.NUM_PARALLEL, 1])
        final_rewards   = torch.zeros([self.NUM_PARALLEL, 1])

        obs = self.envs.reset()
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()
        current_obs = obs
        R_base = []
        for step in range(self.max_step):
            with torch.no_grad():
                # action = actor_critic.act(rollouts.observations[step]) # ここでアクション決めて
                action = torch.zeros(self.NUM_PARALLEL, self.NUM_AGENTS).long().to(self.device) # 各観測に対する，各エージェントの行動
                print("obs",obs)
                for i, actor_critic in enumerate( actor_critics ):
                    # print(actor_critic.__class__.__name__)
                    tmp_action = actor_critic.act_greedy(obs) # ここでアクション決めて
                    action[:,i] = tmp_action.squeeze()
                print("action",action)
            obs, reward, done, infos = self.envs.step(action) # これで時間を進める
            episode_rewards += reward
            R_base.append(reward.item())
            # if done then clean the history of observation
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            if DEBUG: print("done.shape",done.shape)
            if DEBUG: print("masks.shape",masks.shape)
            if DEBUG: print("obs.shape",obs.shape)
            with open(self.resdir + "/episode_reward.txt", "a") as f:
                for i, info in enumerate(infos):
                    if 'episode' in info:
                        f.write("{:}\t{:}\n".format(info['env_id'], info['episode']['r']))
                        print(info['env_id'], info['episode']['r'])

            # イベント保存のためには，->要仕様検討
            with open(self.resdir + "/event.txt", "a") as f: 
                for i, info in enumerate(infos):
                    if 'events' in info:
                        for event in info['events']:
                            f.write("{:}\n".format(event))
                            # print(event)
                        # episode[i] += 1

            final_rewards *= masks
            final_rewards += (1-masks) * episode_rewards

            episode_rewards *= masks
            current_obs     *= masks

            current_obs = obs # ここで観測を更新している
            with open(self.resdir + "/reward_log.txt", "a") as f:
                f.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format(step, reward.max().numpy(), reward.min().numpy(), reward.mean().numpy(), episode_rewards.max().numpy(), episode_rewards.min().numpy(), episode_rewards.mean().numpy()))
                print(step, reward.mean().numpy(), episode_rewards.mean().numpy())
        print("ここでtest終了")
        return final_rewards.mean().numpy(), R_base

def save_model(model, fn="model"):
    torch.save(model.state_dict(), fn)

def load_model(n_in, n_out, fn="model"):
    model = ActorCritic(n_in, n_out)
    model.load_state_dict(torch.load(fn))
    model.eval()
    return model

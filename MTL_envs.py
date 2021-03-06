import gym
import torch
import copy
# from model import ActorCritic
# from brain import Brain
from MTL_model import ActorCritic
from MTL_brain import Brain
from storage import RolloutStorage
from controler import FixControler
from mp.envs import make_vec_envs
import json
import numpy as np
import pandas as pd
import configparser
from torch.autograd import Variable
import shutil
import os, sys, glob
from edges import Edge
import datetime

DEBUG = False # True # False # True # False # True # False # True # False

class Curriculum:
    def run(self, args):
        config = configparser.ConfigParser()
        config.read(args.configfn)
        test_env = Environment(args, "test")
        # training_targets = list( np.loadtxt( config['TRAINING']['training_target'] , dtype=int ) )
        shelters = np.loadtxt( config['SIMULATION']['actionfn'] , dtype=int )
        edgedir = config['SIMULATION']['edgedir'] # datadir の代用
        edges = Edge(edgedir) # 暫定
        dt = datetime.datetime.now() # 現在時刻->実験開始時刻をログ出力するため
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

        # better_agents = [] # ルールベースを超えたエージェントIDのリスト
        # dict_best_model = {}

        dict_FixControler = {}

        update_score = np.zeros(test_env.n_out) # 学習が進んでいるか評価するため
        if args.checkpoint:
            # モデルを読み込む処理
            ifn = args.inputfn
            # ifns = glob.glob(args.inputfn + "_*")
            actor_critic = load_model(test_env.n_in, test_env.n_out, ifn).to(test_env.device)
            actor_critic.set_edges(edges)
            # for ifn in ifns:
            #     print("loading: ",ifn)
            #     node_id = int( ifn.split("_")[-1] )
            #     actor_critic = load_model(test_env.n_in, test_env.n_out, ifn).to(test_env.device)
            #     actor_critic.set_edges(edges)
            #     dict_model[node_id] = actor_critic
            scorefn = ifn + ".score"
            if os.path.exists(scorefn):# 各エージェントの学習結果の最高性能の記録
                update_score = pd.read_pickle(scorefn)
                print("update_score", update_score)
            betterfn = ifn + ".better_agents"
            if os.path.exists(scorefn):# 各エージェントの学習結果の最高性能の記録
                better_agents = pd.read_pickle(betterfn)
                actor_critic.set_better_agents(better_agents)
                print("better_agents", better_agents)
        else:
            actor_critic = ActorCritic(test_env.n_in, test_env.n_out)
            actor_critic.set_edges(edges)

        # dict_FixControlerに，ルールベースのエージェントを配置しておく
        fix_list = []
        for sid, shelter in enumerate(shelters):
            controler = FixControler(sid, edges)
            # if shelter in dict_model: # モデルを読み込んだnodeはスキップ
            #     continue
            # dict_FixControler[shelter] = controler
            dict_FixControler[sid] = controler
            fix_list.append(sid)
        # sys.exit()
        # if args.test: # testモードなら，以下の学習はしない
        #     print(actor_critic.better_agents)
        #     base_score, R_base = test_env.test(actor_critic, dict_FixControler) # 読み込んだモデルの評価値を取得
            # sys.exit()
        # else:
        base_score, R_base = test_env.test(actor_critic, dict_FixControler, test_list=[], fix_list=fix_list) # ルールベースの評価値を取得
        T_open, travel_time = R_base
        print("初回のスコア", base_score, T_open, np.mean(travel_time))
        R_base = (T_open , travel_time) # train環境に入力するため
        with open(resdir + "/Curriculum_log.txt", "a") as f:
            f.write("Curriculum start: " + dt.strftime('%Y年%m月%d日 %H:%M:%S') + "\n")
            f.write("initial score:\t{:}\n".format(base_score))
            print("initial score:\t{:}\n".format(base_score))
        best_score = copy.deepcopy(base_score) # 初回を暫定一位にする

        if args.test: # testモードなら，以下の学習はしない
            sys.exit()

        # dict_best_model = copy.deepcopy(dict_model)
        # tmp_fixed = copy.deepcopy(dict_target["training"])
        loop_i = 0 # カリキュラムのループカウンタ
        NG_target = [] # scoreが改善しなかったtargetリスト
        train_env = Environment(args, "train", R_base, loop_i)
        # while True:
        while (loop_i == 0):
            loop_i += 1
            flg_update = False
            # for training_target in training_targets:
            for training_target in range(actor_critic.n_out):
                if training_target in NG_target: # 改善しなかった対象は省略
                    continue
                actor_critic = train_env.train(actor_critic, dict_FixControler, config, training_target)
                tmp_score, _ = test_env.test(actor_critic, dict_FixControler, test_list=[training_target])
                with open(resdir + "/Curriculum_log.txt", "a") as f:
                    f.write("{:}\t{:}\t{:}\t{:}\n".format(loop_i, train_env.NUM_EPISODES, training_target, tmp_score))
                    print(loop_i, training_target, tmp_score)

                # 過去最高の性能を更新したエージェントがいれば，学習を継続する
                if update_score[training_target] == 0 or tmp_score < update_score[training_target]:
                    flg_update = True
                    NG_target = []
                    update_score[training_target] = copy.deepcopy( tmp_score )

                if tmp_score < base_score: # scoreは移動時間なので小さいほどよい
                    # best_score = copy.deepcopy(tmp_score)
                    if training_target not in actor_critic.better_agents:
                        actor_critic.better_agents.append(training_target)
                        print("better_agents", actor_critic.better_agents)

                if tmp_score < best_score: # scoreは移動時間なので小さいほどよい
                    best_score = copy.deepcopy(tmp_score)
                else: # 性能を更新できなかったら，NG_targetに記録
                    NG_target.append(training_target)
                if args.save: # 毎回モデルを保存
                    save_model(actor_critic, resdir + '/' + outputfn)
                    pd.to_pickle(update_score, resdir + '/' + outputfn + ".score")
                    pd.to_pickle(actor_critic.better_agents, resdir + '/' + outputfn + ".better_agents")
            if not flg_update: # 1個もtargetが更新されなかったら終了
                break
        # 終了
        dt = datetime.datetime.now() # 現在時刻->実験開始時刻をログ出力するため
        with open(resdir + "/Curriculum_log.txt", "a") as f:
            f.write("Curriculum 正常終了: " + dt.strftime('%Y年%m月%d日 %H:%M:%S') + "\n")
            f.write("final score:\t{:}\n".format(best_score))
            print("ここでCurriculum終了")
            print("final score:\t{:}\n".format(best_score))

class Environment:
    def __init__(self, args, flg_test=False, R_base=(None,None), loop_i=999):
        config = configparser.ConfigParser()
        config.read(args.configfn)
        # config.read('config.ini')
        self.sim_time  = config.getint('SIMULATION', 'sim_time')
        self.interval  = config.getint('SIMULATION', 'interval')
        self.max_step  = int( np.ceil( self.sim_time / self.interval ))
        self.loop_i = loop_i
        # NUM_PROCESSES     = config.getint('TRAINING', 'num_processes')
        if flg_test=="test":
            self.NUM_PARALLEL = 1
            print(config['TEST'])
            self.resdir = config['TEST']['resdir']
        else: # "training"
            self.NUM_PARALLEL     = config.getint('TRAINING', 'num_parallel')
            print(config['TRAINING'])
            self.resdir = config['TRAINING']['resdir']
        if not os.path.exists(self.resdir):
            os.makedirs(self.resdir)

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

        # print(resdir)
        # shutil.copy2(args.configfn, self.resdir)
        # with open(self.resdir + "/args.txt", "w") as f:
        #     json.dump(args.__dict__, f, indent=2)

        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        self.envs = make_vec_envs(args.env_name, args.seed, self.NUM_PARALLEL, self.device, self.datadirs, config, R_base)
        self.n_in  = self.envs.observation_space.shape[0]
        self.n_out = self.envs.action_space.n
        self.obs_shape       = self.n_in
        self.obs_init = self.envs.reset()

    # def set_R_base(self, R_base):
    #     self.envs.set_R_base(R_base)

    def train(self, actor_critic, dict_FixControler, config, training_target):
        self.NUM_AGENTS = actor_critic.n_out
        # self.NUM_AGENTS = len(dict_model)
        # print("train", dict_model)
        # actor_critics = []
        # local_brains = []
        # rollouts = []
        # actor_critic = dict_model[training_target]
        global_brain = Brain(actor_critic, config)
        rollout = RolloutStorage(self.NUM_ADVANCED_STEP, self.NUM_PARALLEL, self.obs_shape, self.device)

        current_obs     = torch.zeros(self.NUM_PARALLEL, self.obs_shape).to(self.device)
        episode_rewards = torch.zeros([self.NUM_PARALLEL, 1])
        final_rewards   = torch.zeros([self.NUM_PARALLEL, 1])

        episode         = np.zeros(self.NUM_PARALLEL)

        # obs = self.envs.reset()
        obs = self.obs_init
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()
        current_obs = obs

        rollout.observations[0].copy_(current_obs)

        while True:
            for step in range(self.NUM_ADVANCED_STEP):
                if DEBUG: agent_type = []
                with torch.no_grad():
                    # action = actor_critic.act(rollouts.observations[step]) # ここでアクション決めて
                    action = torch.zeros(self.NUM_PARALLEL, self.NUM_AGENTS).long().to(self.device) # 各観測に対する，各エージェントの行動
                    if DEBUG: print("actionサイズ",self.NUM_PARALLEL, self.NUM_AGENTS)
                    for i in range( actor_critic.n_out ):
                        if i == training_target:
                            tmp_action = actor_critic.act(current_obs, i)
                            target_action = copy.deepcopy(tmp_action)
                            if DEBUG: agent_type.append("actor_act")
                        elif i in actor_critic.better_agents:
                            # print(actor_critic.__class__.__name__)
                            tmp_action = actor_critic.act_greedy(obs, i) # ここでアクション決めて
                            if DEBUG: agent_type.append("actor_greedy")
                        else:
                            tmp_action = dict_FixControler[i].act_greedy(obs)
                            if DEBUG: agent_type.append("FixControler")
                        action[:,i] = tmp_action.squeeze()
                    # for i, (k,v) in enumerate( dict_model.items() ):
                    #     if k == training_target:
                    #         tmp_action = v.act(current_obs)
                    #         target_action = copy.deepcopy(tmp_action)
                    #     else:
                    #         tmp_action = v.act_greedy(current_obs)
                    #     action[:,i] = tmp_action.squeeze()
                if DEBUG: print(agent_type)
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
                with open(self.resdir + "/reward_log.txt", "a") as f: # このログはエピソードが終わったときだけでいい->要修正
                    f.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format(self.loop_i,training_target, episode.mean(), step, reward.max().numpy(), reward.min().numpy(), reward.mean().numpy(), episode_rewards.max().numpy(), episode_rewards.min().numpy(), episode_rewards.mean().numpy()))
                    print(self.loop_i,training_target, episode.mean(), step, reward.mean().numpy(), episode_rewards.mean().numpy())

            with torch.no_grad():
                next_value = actor_critic.get_value(rollout.observations[-1], training_target).detach()

            rollout.compute_returns(next_value, self.gamma)
            value_loss, action_loss, total_loss, entropy = global_brain.update(rollout, training_target)

            with open(self.resdir + "/loss_log.txt", "a") as f:
                f.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format(self.loop_i, training_target, episode.mean(), value_loss, action_loss, entropy, total_loss))
                print("value_loss {:.4f}\taction_loss {:.4f}\tentropy {:.4f}\ttotal_loss {:.4f}".format(value_loss, action_loss, entropy, total_loss))

            rollout.after_update()
            
            if int(episode.mean())+1 > self.NUM_EPISODES:
                # print("ループ抜ける")
                break
        # ここでベストなモデルを保存していた（備忘）
        print("%s番目のエージェントのtrain終了"%training_target)
        # dict_model[training_target] = actor_critic # {}
        return actor_critic

    # def test(self, dict_model): # 1並列を想定する
    def test(self, actor_critic, dict_FixControler, test_list=[], fix_list=[]): # 1並列を想定する
        # self.NUM_AGENTS = len(dict_model)
        self.NUM_AGENTS = actor_critic.n_out
        NUM_PARALLEL = 1
        # actor_critics = []
        # for i, training_target in enumerate( training_targets ):
        # for i, actor_critic in sorted( dict_model.items() ):
        #     actor_critics.append(actor_critic)

        current_obs     = torch.zeros(self.NUM_PARALLEL, self.obs_shape).to(self.device)
        episode_rewards = torch.zeros([self.NUM_PARALLEL, 1])
        final_rewards   = torch.zeros([self.NUM_PARALLEL, 1])

        # obs = self.envs.reset()
        obs = self.obs_init
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()
        current_obs = obs
        T_open = []
        for step in range(self.max_step):
            if DEBUG: agent_type = []
            with torch.no_grad():
                # action = actor_critic.act(rollouts.observations[step]) # ここでアクション決めて
                action = torch.zeros(self.NUM_PARALLEL, self.NUM_AGENTS).long().to(self.device) # 各観測に対する，各エージェントの行動
                print("obs",obs)
                # for i, actor_critic in enumerate( actor_critics ):
                for i in range( actor_critic.n_out ):
                    if ((i in actor_critic.better_agents) or (i in test_list)) and (i not in fix_list):
                        # print(actor_critic.__class__.__name__)
                        tmp_action = actor_critic.act_greedy(obs, i) # ここでアクション決めて
                        if DEBUG: agent_type.append("actor_greedy")
                    else:
                        tmp_action = dict_FixControler[i].act_greedy(obs)
                        if DEBUG: agent_type.append("FixControler")
                    action[:,i] = tmp_action.squeeze()
                print("action",action)
            if DEBUG: print(agent_type)
            obs, reward, done, infos = self.envs.step(action) # これで時間を進める
            # episode_rewards += reward
            T_open.append(reward.item())
            # if done then clean the history of observation
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            if DEBUG: print(masks)
            # テストのときは，rewardの保存は不要では？
            # if DEBUG: print("done.shape",done.shape)
            # if DEBUG: print("masks.shape",masks.shape)
            # if DEBUG: print("obs.shape",obs.shape)
            # with open(self.resdir + "/episode_reward.txt", "a") as f:
            #     for i, info in enumerate(infos):
            #         if 'episode' in info:
            #             f.write("{:}\t{:}\n".format(info['env_id'], info['episode']['r']))
            #             print(info['env_id'], info['episode']['r'])

            # イベント保存のためには，->要仕様検討
            if 'events' in infos[0]: # test()では１並列前提
                eventsfn = self.resdir + "/event.txt"
                with open(eventsfn, "a") as f: 
                    if DEBUG: print("{:}保存します".format(eventsfn))
                    # for i, info in enumerate(infos):
                    for event in infos[0]['events']:
                        f.write("{:}\n".format(event))
                        if DEBUG: print(event)
                        # episode[i] += 1
            if 'travel_time' in infos[0]: # test()では１並列前提
                travel_time = infos[0]['travel_time']
                if ("all_reached" in infos[0]) and (infos[0]["all_reached"] == True) :
                    ret = np.mean(travel_time)
                else:
                    ret = np.inf
            # final_rewards *= masks
            # final_rewards += (1-masks) * episode_rewards
            # episode_rewards *= masks
            # current_obs     *= masks
            # current_obs = obs # ここで観測を更新している

            # テストのときは，rewardの保存は不要では？
            # with open(self.resdir + "/reward_log.txt", "a") as f:
            #     f.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format(step, reward.max().numpy(), reward.min().numpy(), reward.mean().numpy(), episode_rewards.max().numpy(), episode_rewards.min().numpy(), episode_rewards.mean().numpy()))
            #     print(step, reward.mean().numpy(), episode_rewards.mean().numpy())
            # 逆に，テスト結果をどこかに保存する必要がある

        print("ここでtest終了")
        return ret, (T_open, travel_time)
        # return final_rewards.mean().numpy(), (T_open, travel_time)

def save_model(model, fn="model"):
    torch.save(model.state_dict(), fn)

def load_model(n_in, n_out, fn="model"):
    model = ActorCritic(n_in, n_out)
    model.load_state_dict(torch.load(fn))
    # model.eval()
    return model

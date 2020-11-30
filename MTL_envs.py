import gym
import torch
import copy
# from model import ActorCritic
# from brain import Brain
from MTL_model import ActorCritic
from MTL_model import ActorN_CriticN_share0 as ActorN_CriticN
from MTL_brain import Brain
from storage import RolloutStorage
# from controler import FixControler
from mp.envs import make_vec_envs
from converter import RewardMaker
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


        update_score = np.zeros(test_env.n_out) # 学習が進んでいるか評価するため
        if args.checkpoint:
            # モデルを読み込む処理
            ifn = args.inputfn
            # ifns = glob.glob(args.inputfn + "_*")
            actor_critic = load_model(test_env.n_in, test_env.n_out, ifn).to(test_env.device)

            # 下記のパラメータはload_modelで設定されない
            actor_critic.set_edges(edges)
            actor_critic.set_FixControler(edges)

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

            # betterfn = ifn + ".better_agents"
            # if os.path.exists(scorefn):# 各エージェントの学習結果の最高性能の記録
            #     better_agents = pd.read_pickle(betterfn)
            #     actor_critic.set_better_agents(better_agents)
            #     print("better_agents", better_agents)

        else:
            # actor_critic = ActorCritic(test_env.n_in, test_env.n_out)
            actor_critic = ActorN_CriticN(test_env.n_in, test_env.n_out)
            actor_critic.set_edges(edges)
            actor_critic.set_FixControler(edges)

        # dict_FixControler = {}
        # dict_FixControlerに，ルールベースのエージェントを配置しておく
        fix_list = []
        for sid, shelter in enumerate(shelters):
            fix_list.append(sid)
            # controler = FixControler(sid, edges)
            # if shelter in dict_model: # モデルを読み込んだnodeはスキップ
            #     continue
            # dict_FixControler[shelter] = controler
            # dict_FixControler[sid] = controler
        # sys.exit()
        # if args.test: # testモードなら，以下の学習はしない
        #     print(actor_critic.better_agents)
        #     base_score, R_base = test_env.test(actor_critic, dict_FixControler) # 読み込んだモデルの評価値を取得
            # sys.exit()
        # else:
        if args.test:
            # base_score, R_base = test_env.test(actor_critic, dict_FixControler, test_list=[], fix_list=[]) # 学習結果の評価値を取得
            base_score, R_base = test_env.test(actor_critic, test_list=[], fix_list=[]) # 学習結果の評価値を取得
        else:
            # base_score, R_base = test_env.test(actor_critic, dict_FixControler, test_list=[], fix_list=fix_list) # ルールベースの評価値を取得
            base_score, R_base = test_env.test(actor_critic, test_list=[], fix_list=fix_list) # ルールベースの評価値を取得
        # T_open, dict_travel_time = R_base
        # print("初回のスコア", base_score, T_open) # , np.mean(travel_time))
        print("初回のスコア", base_score) # , np.mean(travel_time))
        # R_base = (T_open , dict_travel_time) # train環境に入力するため
        with open(resdir + "/Curriculum_log.txt", "a") as f:
            f.write("Curriculum start: " + dt.strftime('%Y年%m月%d日 %H:%M:%S') + "\n")
            f.write("initial score:\t{:}\n".format(base_score))
            print("initial score:\t{:}\n".format(base_score))
        best_score = copy.deepcopy(base_score) # 初回を暫定一位にする

        if args.test or args.base: # testモードなら，以下の学習はしない
            # baseモードは，ベースラインを実行するだけで終了
            sys.exit()

        training_targets = [3,4,13,14,15,16,17,18] # 最初に向かう人数が定員以上の避難所
        training_targets = list( range(actor_critic.n_out) )
        # capa_over_shelter_ids = [17,18] # 最初に向かう人数が定員以上の避難所
        # dict_best_model = copy.deepcopy(dict_model)
        # tmp_fixed = copy.deepcopy(dict_target["training"])
        # loop_i = 0 # カリキュラムのループカウンタ
        # NG_target = [] # scoreが改善しなかったtargetリスト
        # train_env = Environment(args, "train", R_base, loop_i)
        train_env = Environment(args, "train")
        train_env.reward_maker.set_R_base(R_base)
        while True:
        # while (loop_i == 0):
            actor_critic.update_eps(actor_critic.eps - 0.01)

            train_one_by_one = False

            if train_one_by_one: # エージェントを1個ずつ学習する
                # flg_update = False
                for training_target in training_targets:
                    actor_critic = train_env.train(actor_critic, config, training_target)
                    tmp_score, _ = test_env.test(actor_critic, test_list=[training_target])
                    with open(resdir + "/Curriculum_log.txt", "a") as f:
                        f.write("{:}\t{:}\t{:}\t{:}\n".format(actor_critic.eps, train_env.NUM_EPISODES, training_target, tmp_score))
                        print(actor_critic.eps, training_target, tmp_score)

                    # 過去最高の性能を保持する
                    if update_score[training_target] == 0 or tmp_score < update_score[training_target]:
                        update_score[training_target] = copy.deepcopy( tmp_score )

                    if args.save: # 毎回モデルを保存
                        save_model(actor_critic, resdir + '/' + outputfn)
                        pd.to_pickle(update_score, resdir + '/' + outputfn + ".score")
                best_score = min(update_score)
                break

            else: # 全エージェントをまとめて学習する
                actor_critic = train_env.train(actor_critic, config)
                tmp_score, _ = test_env.test(actor_critic, test_list=training_targets)
                with open(resdir + "/Curriculum_log.txt", "a") as f:
                    f.write("{:}\t{:}\t{:}\t{:}\n".format(actor_critic.eps, train_env.NUM_EPISODES, -1, tmp_score))
                    print(actor_critic.eps, -1, tmp_score)

                if args.save: # 毎回モデルを保存
                    save_model(actor_critic, resdir + '/' + outputfn)
                best_score = min(tmp_score, best_score)
                break

        # 終了
        dt = datetime.datetime.now() # 現在時刻->実験開始時刻をログ出力するため
        with open(resdir + "/Curriculum_log.txt", "a") as f:
            f.write("Curriculum 正常終了: " + dt.strftime('%Y年%m月%d日 %H:%M:%S') + "\n")
            f.write("final score:\t{:}\n".format(best_score))
            print("ここでCurriculum終了")
            print("final score:\t{:}\n".format(best_score))

class Environment:
    # def __init__(self, args, flg_test=False, R_base=(None,None), loop_i=999):
    def __init__(self, args, flg_test=False):
        config = configparser.ConfigParser()
        config.read(args.configfn)
        # config.read('config.ini')
        self.sim_time  = config.getint('SIMULATION', 'sim_time')
        self.interval  = config.getint('SIMULATION', 'interval')
        self.max_step  = int( np.ceil( self.sim_time / self.interval ))
        # self.loop_i = loop_i
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

        # self.envs = make_vec_envs(args.env_name, args.seed, self.NUM_PARALLEL, self.device, self.datadirs, config, R_base)
        self.envs = make_vec_envs(args.env_name, args.seed, self.NUM_PARALLEL, self.device, self.datadirs, config)
        self.n_in  = self.envs.observation_space.shape[0]
        self.n_out = self.envs.action_space.n
        self.obs_shape       = self.n_in
        self.obs_init = self.envs.reset()

        self.reward_maker = RewardMaker()
        self.reward_maker.set_agentfns(self.datadirs)
        self.reward_maker.set_edges(self.datadirs)
        self.m = torch.nn.ReLU()

    def set_R_base(self, R_base):
        self.reward_maker.set_R_base(R_base)
        # self.envs.set_R_base(R_base)

    # def train(self, actor_critic, dict_FixControler, config, training_target):
    def train(self, actor_critic, config, training_target=None):
        self.NUM_AGENTS = actor_critic.n_out
        # self.NUM_AGENTS = len(dict_model)
        # print("train", dict_model)
        # actor_critics = []
        # local_brains = []
        # actor_critic = dict_model[training_target]
        global_brain = Brain(actor_critic, config)
        if training_target is None:
            rollouts = []
            for i in range(self.NUM_AGENTS ):
                rollout = RolloutStorage(self.NUM_ADVANCED_STEP, self.NUM_PARALLEL, self.obs_shape, self.device)
                rollouts.append(rollout)
        else:
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
            # ここでepsを更新すると，ターゲットごとに異なる値を使うことになる．．．
            # actor_critic.loop_i += 1
            # actor_critic.update_eps()
            # actor_critic.eps -= 0.1
            # if actor_critic.eps < 0:
            #     actor_critic.eps = 0
            for step in range(self.NUM_ADVANCED_STEP):
                if DEBUG: agent_type = []
                with torch.no_grad():
                    # action = actor_critic.act(rollouts.observations[step]) # ここでアクション決めて
                    # action = torch.zeros(self.NUM_PARALLEL, self.NUM_AGENTS).long().to(self.device) # 各観測に対する，各エージェントの行動
                    # if DEBUG: print("actionサイズ",self.NUM_PARALLEL, self.NUM_AGENTS)
                    action = actor_critic.act_all(obs, training_target)
                    if training_target is None:
                        target_actions = []
                        for i in range(self.NUM_AGENTS ):
                            target_action = copy.deepcopy( action[:,i].unsqueeze(1) )
                            target_action = action[:,i].unsqueeze(1)
                            target_actions.append(target_action)
                    else:
                        target_action = copy.deepcopy( action[:,training_target].unsqueeze(1) )
                        target_action = action[:,training_target].unsqueeze(1)
                if DEBUG: print(agent_type)
                if DEBUG: print("step前のここ？",action.shape)
                obs, dummy_rewards, dones, infos = self.envs.step(action) # これで時間を進める
                # reward = rewards[training_target] # 学習対象エージェントの報酬だけ取り出す
                # reward, _base, _score, _base2, _score2 = self.reward_maker.info2reward(infos, training_target, step)
                if training_target is None:
                    rewards = []
                    for i in range(self.NUM_AGENTS ):
                        reward1, _base, _score = self.reward_maker.info2rewardWalk(infos, i, step)
                        reward2, _base2, _score2 = self.reward_maker.info2rewardArrive(infos, i, step)
                        reward2 = self.m(reward2)
                        reward = 0.5 * (reward1 + reward2)
                        rewards.append(reward)
                        episode_rewards += reward # 面倒なので全報酬の合計を観察することに．．．
                else:
                    reward1, _base, _score = self.reward_maker.info2rewardWalk(infos, training_target, step)
                    reward2, _base2, _score2 = self.reward_maker.info2rewardArrive(infos, training_target, step)
                    reward2 = self.m(reward2)
                    reward = 0.5 * (reward1 + reward2)
                    episode_rewards += reward

                if DEBUG: print("info2reward", reward)
                # if done then clean the history of observation
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
                if DEBUG: print("done.shape",done.shape)
                if DEBUG: print("masks.shape",masks.shape)
                if DEBUG: print("obs.shape",obs.shape)

                # for i, done in enumerate(dones):
                #     if done:
                #         episode[i] += 1
                #         episode_rewards[i] = 0 # ここで０にするとepisode_reward.txtに０ばかり記録される

                # final_rewards *= masks
                # final_rewards += (1-masks) * episode_rewards

                # episode_rewards *= masks
                # current_obs     *= masks

                # current_obs = obs # ここで観測を更新している

                if training_target is None:
                    for i in range(self.NUM_AGENTS ):
                        rollouts[i].insert(obs, target_actions[i].data, reward, masks, self.NUM_ADVANCED_STEP)
                else:
                    rollout.insert(obs, target_action.data, reward, masks, self.NUM_ADVANCED_STEP)

                with open(self.resdir + "/reward_log.txt", "a") as f: # このログはエピソードが終わったときだけでいい？->報酬による
                    if training_target is None:
                        for i in range(self.NUM_AGENTS ):
                            out = "{:}\t{:}\t{:}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(i, episode.mean(), step, rewards[i].max().numpy(), rewards[i].min().numpy(), rewards[i].mean().numpy())
                            f.write("%s\n"%out)
                        out = "{:}\t{:}\t{:.4f}\t{:.4f}\t{:.4f}".format(episode.mean(), step, episode_rewards.max().numpy(), episode_rewards.min().numpy(), episode_rewards.mean().numpy())
                        f.write("%s\n"%out)
                        print(training_target, episode.mean(), step, episode_rewards.mean().numpy())
                    else:
                        out = "{:}\t{:}\t{:}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(training_target, episode.mean(), step, reward1.max().numpy(), reward1.min().numpy(), reward1.mean().numpy())
                        out += "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(reward2.max().numpy(), reward2.min().numpy(), reward2.mean().numpy(), episode_rewards.max().numpy(), episode_rewards.min().numpy(), episode_rewards.mean().numpy())
                        f.write("%s\n"%out)
                        print(training_target, episode.mean(), step, reward1.mean().numpy(), reward2.mean().numpy(),episode_rewards.mean().numpy())
                        # print( _base, _score, _base2, _score2)
                        # f.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format(self.loop_i,training_target, episode.mean(), step, reward.max().numpy(), reward.min().numpy(), reward.mean().numpy(), episode_rewards.max().numpy(), episode_rewards.min().numpy(), episode_rewards.mean().numpy()))
                        # print(self.loop_i,training_target, episode.mean(), step, reward.mean().numpy(), episode_rewards.mean().numpy())

            if training_target is None:
                for i in range(self.NUM_AGENTS ):
                    with torch.no_grad():
                        next_value = actor_critic.get_value(rollouts[i].observations[-1], i).detach()

                    rollouts[i].compute_returns(next_value, self.gamma)
                    value_loss, action_loss, total_loss, entropy = global_brain.update(rollouts[i], i)
                    rollouts[i].after_update()

                    with open(self.resdir + "/loss_log.txt", "a") as f:
                        f.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format(i, actor_critic.eps, episode.mean(), value_loss, action_loss, entropy, total_loss))
                        print("value_loss {:.4f}\taction_loss {:.4f}\tentropy {:.4f}\ttotal_loss {:.4f}".format(value_loss, action_loss, entropy, total_loss))

                with open(self.resdir + "/episode_reward.txt", "a") as f:
                    for j, info in enumerate(infos):
                        if 'episode' not in info:
                            continue
                        f.write("{:}\t{:}\t{:}\t{:}\n".format(j,episode[j], infos[j]['env_id'], episode_rewards[j]))
                        # print(training_target, episode[j], info['env_id'], info['episode']['r'])
                        print(j, episode[j], infos[j]['env_id'], episode_rewards[j])
                        # episode[i] += 1 # episode_rewards.mean().numpy()

            else:

                with torch.no_grad():
                    next_value = actor_critic.get_value(rollout.observations[-1], training_target).detach()

                rollout.compute_returns(next_value, self.gamma)
                value_loss, action_loss, total_loss, entropy = global_brain.update(rollout, training_target)
                rollout.after_update()

                with open(self.resdir + "/loss_log.txt", "a") as f:
                    f.write("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n".format(training_target, actor_critic.eps, episode.mean(), value_loss, action_loss, entropy, total_loss))
                    print("value_loss {:.4f}\taction_loss {:.4f}\tentropy {:.4f}\ttotal_loss {:.4f}".format(value_loss, action_loss, entropy, total_loss))

                with open(self.resdir + "/episode_reward.txt", "a") as f:
                    for i, info in enumerate(infos):
                        if 'episode' in info:
                            f.write("{:}\t{:}\t{:}\t{:}\n".format(training_target,episode[i], info['env_id'], info['episode']['r']))
                            print(training_target, episode[i], info['env_id'], info['episode']['r'])
                            # episode[i] += 1

            for i, done in enumerate(dones):
                if done:
                    episode[i] += 1
                    episode_rewards[i] = 0 # ループ抜けてから０にする


            if int(episode.mean())+1 > self.NUM_EPISODES:
                print("ループ抜ける")
                break
        if training_target is None:
            print("train終了")
        else:
            print("%s番目のエージェントのtrain終了"%training_target)
        # dict_model[training_target] = actor_critic # {}
        return actor_critic

    # def test(self, dict_model): # 1並列を想定する
    # def test(self, actor_critic, dict_FixControler, test_list=[], fix_list=[]): # 1並列を想定する
    def test(self, actor_critic, test_list=[], fix_list=[]): # 1並列を想定する
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
                action = actor_critic.test_act_all(obs, test_list, fix_list)
                # target_action = copy.deepcopy( action[:,training_target] )
            #     action = torch.zeros(self.NUM_PARALLEL, self.NUM_AGENTS).long().to(self.device) # 各観測に対する，各エージェントの行動
            #     if DEBUG: print("obs",obs)
            #     # for i, actor_critic in enumerate( actor_critics ):
            #     for i in range( actor_critic.n_out ):
            #         if ((i in actor_critic.better_agents) or (i in test_list)) and (i not in fix_list):
            #             # print(actor_critic.__class__.__name__)
            #             tmp_action = actor_critic.act_greedy(obs, i) # ここでアクション決めて
            #             if DEBUG: agent_type.append("actor_greedy")
            #         else:
            #             tmp_action = dict_FixControler[i].act_greedy(obs)
            #             if DEBUG: agent_type.append("FixControler")
            #         action[:,i] = tmp_action.squeeze()
            #     if DEBUG: print("action",action)
            # if DEBUG: print(agent_type)
            obs, reward, dones, infos = self.envs.step(action) # これで時間を進める
            # episode_rewards += reward
            # if done then clean the history of observation
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
            if DEBUG: print(masks)

            # イベント保存のためには，->要仕様検討
            # print(infos[0]["events"])
            if 'events' in infos[0]: # test()では１並列前提
                eventsfn = self.resdir + "/event.txt"
                # with open(eventsfn, "a") as f: 
                with open(eventsfn, "w") as f: 
                    if DEBUG: print("{:}保存します".format(eventsfn))
                    # for i, info in enumerate(infos):
                    for event in infos[0]['events']:
                        f.write("{:}\n".format(event))
                        if DEBUG: print(event)
                        # episode[i] += 1
            # 以下のコメントアウトで，評価指標を切り替える
            ret, dict_travel_times = self.reward_maker.info2traveltime(infos) # 平均移動時間で評価
            # ret, dict_travel_times = self.reward_maker.info2completetime(infos) # 避難完了時間で評価

            T_open.append(infos[0])
            # T_open.append(reward.item())

        print("ここでtest終了")
        return ret[0], (T_open, dict_travel_times[0])
        # return final_rewards.mean().numpy(), (T_open, travel_time)

def save_model(model, fn="model"):
    torch.save(model.state_dict(), fn)
    # print("eps", model.eps)
    pd.to_pickle(model.eps, fn + ".eps")

def load_model(n_in, n_out, fn="model"):
    model = ActorN_CriticN(n_in, n_out)
    # model = ActorCritic(n_in, n_out)
    model.load_state_dict(torch.load(fn))
    eps = pd.read_pickle(fn + ".eps")
    print("eps", eps)
    model.update_eps(eps)
    # model.eval()
    return model

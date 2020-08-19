import argparse
import gym
import numpy as np
from envs import Environment, load_model
from myenv import env
import torch

parser = argparse.ArgumentParser(description="A2C-test")
parser.add_argument('--env-name', type=str, default='simenv-v0', 
        help='Environments')
parser.add_argument('--scenario', type=str, default='N80000r0i0', 
        help='Scenario')
parser.add_argument('--resdir', type=str, default='results', 
        help='Results Directory')
parser.add_argument('--inputfn', type=str, default="model", 
        help='model file path')

if __name__ == '__main__':
    args = parser.parse_args()

    root_path = "mkUserlist/data/"
    datadir   = root_path + args.scenario

    env   = gym.make(args.env_name)
    env.set_datadir(datadir)
    env.set_resdir("result/%s"%args.scenario)
    n_in  = env.observation_space.shape[0]
    n_out = env.action_space.n

    model = load_model(n_in, n_out, args.inputfn)
    model.eval()
    
    state = env.reset()
    state = np.array([state])
    state = torch.from_numpy(state).float()

    total_reward = 0.0
    done = True
    actions = []
    j = 0

    while True:
        # action = model.act(state)
        action = model.act_greedy(state)
        state, reward, done, _ = env.step(action)
        total_reward += env._get_travel_time(state)
        actions.append(int(action[0][0]))
        print(env.interval * j, action, np.sum(state[env.num_edges * (env.obs_step-1):])*env.interval/env.num_agents, total_reward)
        if done:
            with open(args.resdir + "/" + args.scenario, "w") as f:
                f.write("{:}\t{:}\t{:}\n".format(args.scenario, total_reward, actions))
            print(actions)
            print(total_reward)
            actions = []
            break
        state = np.array([state])
        state = torch.from_numpy(state).float()

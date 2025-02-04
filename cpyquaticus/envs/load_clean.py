from faster_envs import Cpyquaticus
import time
import argparse
import gymnasium as gym
import numpy as np
import os
import logging

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import sys

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class DoNothing:
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space
    def get_action_and_value(self, x):
        length = x.shape[0]
        actions = torch.ones((length,))*17
        return actions, None, None, None
class RandomAction:
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space
    def get_action_and_value(self, x):
        length = x.shape[0]
        actions = torch.ones((length,))*action_space.sample()
        return action, None, None, None



def make_env():
    def thunk():
        env = Cpyquaticus(c_load='linux')
        return env

    return thunk


if __name__ == "__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]
    env = Cpyquaticus(c_load='linux')
    obs_space = env.observation_space('agent_0')
    act_space = env.action_space('agent_0')
    policy1 = Agent(obs_space, act_space)
    policy1.load_state_dict(torch.load(path))
    policy2 = Agent(obs_space, act_space)
    policy2.load_state_dict(torch.load(path2))

    obs, _ = env.reset()
    reward = {'agent_0':0,'agent_1':0}
    print("obs: ", obs, " type: ", type(obs['agent_0']))
    print("Length: ", len(obs['agent_0']))
    print("Is it tensor now: ", torch.tensor(obs['agent_0']))
    while True:
        actions = {}
        actions['agent_0'] = 17#policy1.get_action_and_value(torch.tensor(obs['agent_0']))[0]
        actions['agent_1'] = 17#policy2.get_action_and_value(torch.tensor(obs['agent_1']))[0]
        obs, rew, term, _, _ = env.step(actions)
        reward['agent_0'] += rew['agent_0']
        reward['agent_1'] += rew['agent_1']
        if term['agent_0']:
            break
    print("Final rewards: ", reward)


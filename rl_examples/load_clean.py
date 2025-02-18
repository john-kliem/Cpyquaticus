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

from clean_ppo import PPOAgent

import time

def make_env():
    def thunk():
        env = Cpyquaticus(c_load='mac')
        return env

    return thunk


if __name__ == "__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]
    env = Cpyquaticus(c_load='mac', render_mode='human', num_steps=600)
    obs_space = env.observation_space('agent_0')
    act_space = env.action_space('agent_0')
    policy1 = PPOAgent(obs_space, act_space)
    policy1.load_state_dict(torch.load(path))
    policy2 = PPOAgent(obs_space, act_space)
    policy2.load_state_dict(torch.load(path2))

    obs, _ = env.reset()
    reward = {'agent_0':0,'agent_1':0}
    print("obs: ", obs, " type: ", type(obs['agent_0']))
    print("Length: ", len(obs['agent_0']))
    print("Is it tensor now: ", torch.tensor(obs['agent_0']))
    step = 0
    while True:
        actions = {}
        actions['agent_0'] = policy1.get_action_and_value(torch.tensor(obs['agent_0']))[0]
        actions['agent_1'] = policy2.get_action_and_value(torch.tensor(obs['agent_1']))[0]
        obs, rew, term, _, _ = env.step(actions)
        if not rew['agent_0'] == 0.0 or not rew['agent_1']==  0.0:
            print("Step: ", step, " Rew: ", rew)
        reward['agent_0'] += rew['agent_0']
        reward['agent_1'] += rew['agent_1']
        if term['agent_0']:
            break
        step += 1
        # time.sleep(0.25)
    print("Final rewards: ", reward)


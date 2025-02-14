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



class DoNothing:
    def __init__(self, observation_space, action_space, device):
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        

    def get_value(self, x):
        return None
    def get_action_and_value(self, obs, action=None):
        return torch.ones((obs.shape[0])) * 17
    def get_action_and_value_train(self, obs, done, action=None):
        return self.get_action_and_value(obs, action), None, None, None
    def set_reward(self, reward):
        return 0
    def anneal_lr(self, iteration, num_iterations):
        return
    def bootstrap(self, num_steps, next_obs, next_done):
        return
    def train(self, batch):
        #TODO: Understand Bootstrap
        return 
    def get_plotting_vars(self, ):
        return 0, 0, 0, 0, 0, 0, 0
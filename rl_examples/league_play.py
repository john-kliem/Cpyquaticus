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
from multi_env.dict_based_sequential import SequentialMultiEnv
from algos.clean_ppo import PPO, PPOAgent, PPOArgs
from cpyquaticus.envs.c_pyquaticus import Cpyquaticus
from cpyquaticus.base_policies.BasePolicies import DoNothing, Random
import random

def make_cpyquaticus_env():
    def thunk():
        env = Cpyquaticus(num_steps=600,c_load='linux')
        return env
    return thunk()

#Given a path to a cleanRL nn file load in the agent
def get_agent(path, algo):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_cpyquaticus_env()
    if "ppo" == algo:
        #Load in PPO NN Algorithm
        algo = PPO(PPOAgent, PPOArgs,env.observation_space('agent_0'), env.action_space('agent_0'), device=device).load(path)
    elif "dqn" == algo:
        #Load in DQN NN Algorithm
        #TODO add DQN Support
        return None
    elif "DoNothing" == algo:
        algo = DoNothing(env.observation_space('agent_0'), env.action_space('agent_0'), device=device)
    elif "Random" == algo:
        algo = Random(env.observation_space('agent_0'), env.action_space('agent_0'), device=device)
    return algo

class MCTFLeague:
    #League Currently Only Supports single agent teams
    def __init__(self, num_main=1):
        #TODO: Reevaluate for inclusion of multi-agent teams
        self.num_main = num_main
        self.num_exploiters = num_main#num_exploiters

        self.main_red = [None for i in range(num_main)]
        self.main_elos = [500 for i in range(num_main)]

        self.exploiters_red = [None for i in range(self.num_exploiters)]
        self.exploiters_red_elos = [500 for i in range(self.num_exploiters)]

        self.prev_reds = []
        self.prev_reds_elos = []

        #Initialize Main Blue Agents
        self.main_blue = [None for i in range(num_main)]
        self.main_elos = [500 for i in range(num_main)]

        self.exploiters_blue = [None for i in range(self.num_exploiters)]
        self.exploiters_blue_elos = [500 for i in range(self.num_exploiters)]

        self.prev_blues = []
        self.prev_blues_elos = []


        self.selfplay_loops = 0
    def train(self, selfplay_loops=1):
        for loop in range(selfplay_loops):
            #Train Main Agents
            for ma in range(self.num_main):
                #Select Red Agents to train against if any
                opponents = {'agents':[], 'percentages':[]}
                #Train Main Blue Agent
                if self.main_blue[ma] == None:
                    #Initialize Main agent
                    opponents = {'agents':['DoNothing',], 'percentages':[1.0,]}
                else:
                    #Assign Opponents
                    opponents = {'agents':[self.main_red[ma],self.exploiter_red[ma], self.prev_reds[random.randint(0,len(self.prev_reds))]], 'percentages':[0.6, 0.25, 0.15]}
                algo = self.main_blue[ma]
                self.main_blue[ma] = self._train('main_blue_'+str(ma), algo, opponents)
                #Train Main Red Agent
                if self.main_red[ma] == None:
                    opponents = {'agents':['DoNothing',], 'percentages':[1.0,]}
                else:
                    opponents = {'agents':[self.main_blue[ma], self.exploiter_blue[ma], self.prev_blues[random.randint(0,len(self.prev_blues))],], 'percentages':[0.6, 0.25, 0.15]}
                algo = self.main_red[ma]
                
                    #Assign Opponents
                self.main_red[ma] = self._train('main_red_'+str(ma), algo, opponents)

                #Train Blue Exploiter
                opponents = {'agents':[self.main_blue[ma],], 'percentages':[1.0,]}
                self.exploiter_blue[ma] = self._train('blue_exploiter_'+str(ma), None, opponents, steps=25000000)
                #Train Red Exploiter
                opponents = {'agents':[self.main_red[ma],], 'percentages':[1.0,]}
                self.exploiter_red[ma] = self._train('red_exploiter_'+str(ma), None, opponents, steps=25000000)
                
            #Evaluate and update elo scores for all agents
                    
        return
    def _train(self,name, algo, opponents_dict, steps=1000000, num_envs=100):
        #Split the number of agents based on the rollout sizes
        #TODO: Make this configurable
        #Slightly modified cleanRL training loop
        indicies = []
        for i, percent in enumerate(opponents_dict['percentages']):
            if i == 0:
                indicies.append((0,int(num_envs*percent)))
            else:
                indicies.append((int(num_envs*opponents_dict['percentages'][i-1]),int(num_envs*percent)))

        
        
        #Save Current Model
        path = f'./Leauge/{name}_{self.selfplay_loops}.pt'
        algo.save(path)
        #Save NN checkpoint path
        return path
if __name__ == "__main__":
    league = MCTFLeague()
    league.train()


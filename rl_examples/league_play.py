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
        env = Cpyquaticus(num_steps=600,c_load='mac')
        return env
    return thunk()

#Given a path to a cleanRL nn file load in the agent
def get_agent(path, algArgs, obs, act):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if path == None:
        algo = PPO(PPOAgent, algArgs,  obs, act, device=device)
    elif "PPO" in path:
        algo = PPO(PPOAgent, algArgs, obs, act, device=device)
        algo.load(path)
    elif "DoNothing" == path:
        algo = DoNothing(obs, act, device=device)
    elif "Random" == path:
        algo = Random(obs, act, device=device)
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
                    if len(self.prev_reds) > 0:
                        opponents = {'agents':[self.main_red[ma],self.exploiters_red[ma], self.prev_reds[random.randint(0,len(self.prev_reds)-1)]], 'percentages':[0.6, 0.25, 0.15]}
                    else:
                        opponents = {'agents':[self.main_red[ma],self.exploiters_red[ma]], 'percentages':[0.80, 0.20]}

                algo = self.main_blue[ma]
                if not self.main_blue[ma] == None:
                    self.prev_blues.append(self.main_blue[ma])
                self.main_blue[ma] = self._train('main_blue_'+str(ma), algo, opponents)
                #Train Main Red Agent
                if self.main_red[ma] == None:
                    opponents = {'agents':['DoNothing',], 'percentages':[1.0,]}
                else:
                    if len(self.prev_blues) > 0:
                        opponents = {'agents':[self.main_blue[ma], self.exploiters_blue[ma], self.prev_blues[random.randint(0,len(self.prev_blues)-1)],], 'percentages':[0.6, 0.25, 0.15]}
                    else:
                        opponents = {'agents':[self.main_blue[ma],self.exploiters_blue[ma]], 'percentages':[0.80, 0.20]}

                algo = self.main_red[ma]
                if not self.main_red[ma] == None:
                    self.prev_reds.append(self.main_red[ma])
                #Assign Opponents
                self.main_red[ma] = self._train('main_red_'+str(ma), algo, opponents)
                #Train Blue Exploiter
                opponents = {'agents':[self.main_blue[ma],], 'percentages':[1.0,]}
                self.exploiters_blue[ma] = self._train('blue_exploiter_'+str(ma), None, opponents)
                #Train Red Exploiter
                opponents = {'agents':[self.main_red[ma],], 'percentages':[1.0,]}
                self.exploiters_red[ma] = self._train('red_exploiter_'+str(ma), None, opponents)
                #Evaluate and update elo scores for all agents
            self.selfplay_loops += 1
        return
    # def elo_scores(self, ):

    def _train(self,name, algo, opponents_dict, steps=1000000, num_envs=100):
        #Split the number of agents based on the rollout sizes
        #TODO: Make this configurable to multi-agent scenarios
        
        #Set Up Indicies for which opponents to collect training data from
        indicies = []
        for i, percent in enumerate(opponents_dict['percentages']):
            if i == 0:
                indicies.append((0,int(num_envs*percent)-1))
            else:
                start_ind = indicies[i-1][1]
                indicies.append((start_ind,start_ind + int(num_envs*percent)-1))
        envs = SequentialMultiEnv([make_cpyquaticus_env for i in range(num_envs)])
        alg_args = PPOArgs()
        alg_args.num_envs = num_envs
        alg_args.num_steps = 1200
        alg_args.batch_size = int(num_envs* alg_args.num_steps)
        alg_args.minibatch_size = int(alg_args.batch_size // 16)
        alg_args.update_epochs = 56
        alg_args.total_timesteps = steps
        
        alg_args.num_iterations = alg_args.total_timesteps // alg_args.batch_size
        
        alg_args.seed = 0

        obs = envs.envs[0].observation_space('agent_0')
        act = envs.envs[0].action_space('agent_0')
        opponents = []
        for i, o in enumerate(opponents_dict['agents']):
            opponents.append(get_agent(o, alg_args, obs, act))
        algo = get_agent(algo, alg_args, obs, act)

        if 'red_exploiter' or 'main_blue':
            to_train = 'agent_0'
        else:
            to_train = 'agent_1'
        #print("To Train: ", to_train)
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=alg_args.seed)
        next_done = torch.zeros(alg_args.num_envs)
        for iteration in range(1, alg_args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            algo.anneal_lr(iteration, alg_args.num_iterations)
            #print("Iteration: ", iteration, " Out of: ", alg_args.num_iterations+1)
            for step in range(0, alg_args.num_steps):
                global_step += alg_args.num_envs

                # obs[step] = next_obs
                # dones[step] = next_done

                # ALGO LOGIC: action logic
                actions = []
                with torch.no_grad():
                    agent_actions = []
                    for ind,a in enumerate(['agent_0', 'agent_1']):
                        if step == 0:
                            algo.buffer_step = 0
                        if a == to_train:
                            agent_next_obs  = torch.tensor([d[a] for d in next_obs if a in d])
                            action, logprob, _, value = algo.get_action_and_value_train(agent_next_obs, next_done)
                        else:
                            agent_next_obs  = torch.tensor([d[a] for d in next_obs if a in d])
                            
                            # Torch obs ([100, 16])
                            actions = []
                            for i,opp in enumerate(opponents):
                                our_next_obs = agent_next_obs[indicies[i][0]:indicies[i][1]]
                                temp,_,_,_ = opp.get_action_and_value(our_next_obs)
                                actions.append(temp)
                            actions = torch.cat(actions, dim=0)

                        agent_actions.append(action)    
                
                actions = torch.stack(agent_actions).T

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(actions.cpu().numpy())
                #TODO Update next_done call for correct shapes
                next_done = terminations
                agent_reward  = torch.tensor([d[to_train] for d in reward if to_train in d])
                algo.set_reward(agent_reward)
                agent_terms = torch.tensor([d[to_train] for d in terminations if to_train in d])
                agent_truncs = torch.tensor([d[to_train] for d in truncations if to_train in d])
                next_done = np.logical_or(agent_terms, agent_truncs)
                # next_obs, next_done = torch.Tensor(next_obs), torch.Tensor(next_done)
                # if True in next_done:
                #     for a in args.to_train:
                #         agent_return = agents[a].rewards.sum(dim=0).reshape(1, args.num_envs)
                #         #Average num envs
                #         avg_agent_return = agent_return.mean(dim=1)
                #         if isinstance(envs, gym.vector.SyncVectorEnv):
                #             for info in infos['final_info']:
                #                 if info and 'episode' in info:
                #                     summary_writers[a].add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                #                     summary_writers[a].add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                #         else:
                #             summary_writers[a].add_scalar("charts/episodic_return", avg_agent_return, global_step)
                        # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            # bootstrap value if not done
               
            agent_next_obs  = torch.tensor([d[to_train] for d in next_obs if to_train in d])
            algo.bootstrap(alg_args.num_steps, agent_next_obs, next_done)
            
            # Optimizing the policy and value network
            b_inds = np.arange(alg_args.batch_size)
            clipfracs = []
            for epoch in range(alg_args.update_epochs):
                # print("Epoch Training")
                np.random.shuffle(b_inds)
                for start in range(0, alg_args.batch_size, alg_args.minibatch_size):
                    end = start + alg_args.minibatch_size
                    mb_inds = b_inds[start:end]
                    algo.train(mb_inds)

            # for a in agents:
            #     if a in args.to_train:
            #         explained_var, v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs = agents[a].get_plotting_vars()
            #         # TRY NOT TO MODIFY: record rewards for plotting purposes
            #         summary_writers[a].add_scalar("charts/learning_rate", agents[a].optimizer.param_groups[0]["lr"], global_step)
            #         summary_writers[a].add_scalar("losses/value_loss", v_loss.item(), global_step)
            #         summary_writers[a].add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            #         summary_writers[a].add_scalar("losses/entropy", entropy_loss.item(), global_step)
            #         summary_writers[a].add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            #         summary_writers[a].add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            #         summary_writers[a].add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            #         summary_writers[a].add_scalar("losses/explained_variance", explained_var, global_step)
            #         print("SPS:", int(global_step / (time.time() - start_time)))
            #         summary_writers[a].add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            
        envs.close()
        
        #Save Current Model
        path = f'./League/PPO_{name}_{self.selfplay_loops}.pt'
        algo.save(path)
        #Save NN checkpoint path
        
        return path
if __name__ == "__main__":
    league = MCTFLeague()
    league.train(selfplay_loops=25)
    


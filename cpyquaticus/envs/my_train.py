from faster_envs import Cpyquaticus
import time
import argparse
import gymnasium as gym
import numpy as np
import os
import logging
import pufferlib
import pufferlib.vector
import pufferlib.wrappers
from pettingzoo.test import parallel_api_test

class DoNothing:
    """
    Example wrapper for training against a random policy.

    To use a base policy, insantiate it inside a wrapper like this,
    and call it from self.compute_actions

    See policies and policy_mapping_fn for how policies are associated
    with agents
    """
    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
    def compute_actions(self, ):
        return [-1 for _ in obs_batch], [], {}
from typing import Optional
import copy

def env_creator():
    env = Cpyquaticus(c_load='mac')
    # env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)

import pufferlib.emulation
import pufferlib.wrappers



from pdb import set_trace as T

import numpy as np
import warnings

import gymnasium
import inspect

import pufferlib
import pufferlib.spaces
from pufferlib import utils, exceptions
from pufferlib.environment import set_buffers
from pufferlib.spaces import Discrete, Tuple, Dict
def dtype_from_space(space):
    if isinstance(space, pufferlib.spaces.Tuple):
        dtype = []
        for i, elem in enumerate(space):
            dtype.append((f'f{i}', dtype_from_space(elem)))
    elif isinstance(space, pufferlib.spaces.Dict):
        dtype = []
        for k, value in space.items():
            dtype.append((k, dtype_from_space(value)))
    elif isinstance(space, (pufferlib.spaces.Discrete)):
        dtype = (np.int32, ())
    elif isinstance(space, (pufferlib.spaces.MultiDiscrete)):
        dtype = (np.int32, (len(space.nvec),))
    else:
        dtype = (space.dtype, space.shape)
    return np.dtype(dtype, align=True)

def flatten_space(space):
    if isinstance(space, pufferlib.spaces.Tuple):
        subspaces = []
        for e in space:
            subspaces.extend(flatten_space(e))
        return subspaces
    elif isinstance(space, pufferlib.spaces.Dict):
        subspaces = []
        for e in space.values():
            subspaces.extend(flatten_space(e))
        return subspaces
    else:
        return [space]
def emulate_observation_space(space):
    emulated_dtype = dtype_from_space(space)
    if isinstance(space, pufferlib.spaces.Box):
        return space, emulated_dtype

    leaves = flatten_space(space)
    dtypes = [e.dtype for e in leaves]
    if dtypes.count(dtypes[0]) == len(dtypes):
        dtype = dtypes[0]
        print("Dtype float: ", dtypes)
    else:
        print("Dtype:uint ")
        dtype = np.dtype(np.uint8)

    mmin, mmax = utils._get_dtype_bounds(dtype)
    numel = emulated_dtype.itemsize // dtype.itemsize
    emulated_space = gymnasium.spaces.Box(low=mmin, high=mmax, shape=(numel,), dtype=dtype)
    return emulated_space, emulated_dtype

def emulate_action_space(space):
    if isinstance(space, pufferlib.spaces.Box):
        return space, space.dtype
    elif isinstance(space, (pufferlib.spaces.Discrete, pufferlib.spaces.MultiDiscrete)):
        return space, np.int32

    emulated_dtype = dtype_from_space(space)
    leaves = flatten_space(space)
    emulated_space = gymnasium.spaces.MultiDiscrete([e.n for e in leaves])
    return emulated_space, emulated_dtype
# def make_env(env_id,):
#     def thunk():
def env_creator():
    def thunk():
        env = Cpyquaticus()
        env = pufferlib.wrappers.PettingZooTruncatedWrapper(env=env)
        return pufferlib.emulation.PettingZooPufferEnv(env=env)
    return thunk()

env = pufferlib.wrappers.PettingZooTruncatedWrapper(env=Cpyquaticus())
single_obs_space, dtype = emulate_observation_space(env.observation_space(env.possible_agents[0]))
print("Single_obs: ", single_obs_space, " Dtype: ", dtype)
single_act_space, adtype = emulate_action_space(env.action_space(env.possible_agents[0]))
print("Single_obs: ", single_act_space, " Dtype: ", adtype)
# vec_env = pufferlib.vector.make(env_creator, num_envs=2, num_workers=1, backend = pufferlib.vector.Multiprocessing)

# obs, _ = env.reset()
# print("Possible Agents: ", env.possible_agents)
# print("Obs Space: ", env.observation_space(env.possible_agents[0]))
# print("Action Space: ", env.action_space(env.possible_agents[0]))
# print("Obs: ", obs)

# actions = {'agent_0':17,'agent_1':17}
# while True:
#     obs, rew, term,_,_ = env.step(actions)
    # print("Possible Agents: ", env.possible_agents)
    # print("Obs Space: ", env.observation_space(env.possible_agents[0])," 2 ", env.observation_space(env.possible_agents[1]))
    # print("Action Space: ", env.action_space(env.possible_agents[0]), " 2 ", env.action_space(env.possible_agents[0]))
    # print("Obs: ", obs)
    # print("rew: ", rew)
    # print()
    # # print()
    # if term:
    #     break


# env = nmmo_creator()
# obs, _ = env.reset()
# structured_obs = obs[1].view(env.obs_dtype)
# env = nmmo.Env()
# obs = env.reset()
# print('NMMO observation space:', structured_obs.dtype)
# print('Packed shape:', obs[1].shape)
# for step in range(10):
#    actions = {a: env.action_space(a).sample() for a in env.agents}
#    obs, rewards, dones, infos = env.step(actions)

    # return env#Cpyquaticus(c_load='mac')#env#pufferlib.emulation.PettingZooPufferEnv(env=env)
# if __name__ == "__main__":
#     num_envs = 2  # Number of environments to run in parallel
#     test = env_creator()
#     print(env.observation_space('agent_0'))
#     print(test)
#     print("reset")
#     print(test.reset())
    # Create vectorized environment
    #vec_env = pufferlib.vector.Serial(env_creator, env_args=None, env_kwargs=None, num_envs=num_envs)
    # # vec_env = pufferlib.vector.Serial(env_creator, env_args=None, env_kwargs=None, num_envs=num_envs)

    # # Reset environment
    # observations, infos = vec_env.reset()
    # print("observations: ", observations)
    # # Run a random action loop for testing
    # for _ in range(10):
    #     actions = {agent: vec_env.action_space(agent).sample() for agent in vec_env.envs[0].agents}
        
    #     obs, rewards, dones, truncs, infos = vec_env.step(actions)
        
    #     print("Observations:", obs)
    #     print("Rewards:", rewards)
    #     print("Dones:", dones)
    #     print("Infos:", infos)
        
    #     if all(dones.values()):
    #         break
    
    # # Close the vectorized environment
    # vec_env.close()
# if __name__ == '__main__':
#     #Competitors: reward_config should be updated to reflect how you want to reward your learning agent
    
#     logging.basicConfig(level=logging.ERROR)
#     env = env_creator()
#     obs, _ = env.reset()
#     # vecenv = pufferlib.vector.make(env_creator, num_envs= 100, num_workers=2, backend=pufferlib.vector.Multiprocessing)

    
    

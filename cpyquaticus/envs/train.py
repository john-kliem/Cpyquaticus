from faster_envs import Cpyquaticus
import time

import argparse
import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import sys
import time

from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec, Policy
import os
import logging
class DoNothing(Policy):
    """
    Example wrapper for training against a random policy.

    To use a base policy, insantiate it inside a wrapper like this,
    and call it from self.compute_actions

    See policies and policy_mapping_fn for how policies are associated
    with agents
    """
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return [17 for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv,PettingZooEnv
from typing import Optional
import copy

class ParallelPettingZooWrapper(ParallelPettingZooEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def reset(self, *, seed: Optional[int] = None, return_info=False, options: Optional[dict] = None):
        obs, infos = super().reset(seed=seed, options=options)
        return obs, infos
        # pass empty info just to align with RLlib code
    def render(self):
        return self.par_env.render()


if __name__ == '__main__':
    #Competitors: reward_config should be updated to reflect how you want to reward your learning agent
    
    logging.basicConfig(level=logging.ERROR)

    
    env_creator = lambda config: Cpyquaticus()
    env = ParallelPettingZooWrapper(Cpyquaticus())
    register_env('cpyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
    obs_space = env.observation_space['agent_0']
    act_space = env.action_space['agent_0']
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 'agent_0':
            return "agent-0-policy"
        # if agent_id == 'agent_1':
            # return "agent-1-policy"
        return "do-nothing"
        #elif agent_id == 2 or agent_id == 'agent-2':
            # change this to agent-1-policy to train both agents at once
        #    return "easy-defend-policy"
        #else:
        #    return "easy-attack-policy"
    
    policies = {'agent-0-policy':(None, obs_space, act_space, {}), 
                'agent-1-policy':(None, obs_space, act_space, {}),
                'do-nothing':(DoNothing, obs_space, act_space, {})}
                #Examples of Heuristic Opponents in Rllib Training (See two lines below)
                #'easy-defend-policy': (DefendGen(2, Team.RED_TEAM, 'easy', 2, env.par_env.agent_obs_normalizer), obs_space, act_space, {}),
                #'easy-attack-policy': (AttackGen(3, Team.RED_TEAM, 'easy', 2, env.par_env.agent_obs_normalizer), obs_space, act_space, {})}
    env.close()
    #Not using the Alpha Rllib (api_stack False) 
    ppo_config = PPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False).environment(env='cpyquaticus').env_runners(num_env_runners=10, num_cpus_per_env_runner=1)
    #If your system allows changing the number of rollouts can significantly reduce training times (num_rollout_workers=15)
    ppo_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["agent-0-policy"],)
    algo = ppo_config.build()
    start = 0
    end = 0
    start = time.time()
    for i in range(10000):
        print("Itr: ", i, flush=True)
        algo.train()
        
        if np.mod(i, 500) == 0:
            print("Saving Checkpoint: ", i, 'elapsed Time: ', time.time()-start)
            chkpt_file = algo.save('./ray_test/iter_'+str(i)+'/')
        # break
# start = time.time()
# x = Cpyquaticus()
# for i in range(10000):
# 	obs, info = x.reset()
# 	while True:
# 		# print("Steps: ", x.steps, " Max Steps: ", x.max_steps)
# 		obs, rew, trunc, term, info = x.step({'agent_0':0, 'agent_1':0})
# 		# print(obs)
# 		# break
# 		# print("Should Trunc: ", trunc['agent_0'], ' Term: ', term['agent_0'])
# 		if trunc['agent_0'] or term['agent_0']:
# 			break
# x.close()
# end = time.time()
# print("Elapsed Time for 10k episodes: ", end-start)

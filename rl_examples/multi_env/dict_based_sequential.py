import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium.spaces import Dict

class SequentialMultiEnv(ParallelEnv):
    def __init__(self, envs):
        """
        A wrapper that runs multiple PettingZoo environments sequentially in the same process.
        
        :param env_fns: A list of callables that create PettingZoo environments.
        """
        self.envs = [e() for e in envs]
        self.num_envs = len(self.envs)
        self.env_pos = 0  # Tracks which env is active
        self.agents = self.envs[0].agents  # Assuming all envs have the same agents
        self.possible_agents = self.envs[0].possible_agents

        # Assuming all environments have the same observation & action spaces
        self.observation_spaces = {agent: self.envs[0].observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.envs[0].action_space(agent) for agent in self.possible_agents}
        self.num_steps = self.envs[0].max_steps
        self.resets = []
    def reset(self, seed=None, options=None):
        """Reset all environments and start from the first."""
        observations = []
        infos = []
        for env in self.envs:
            obs, info = env.reset()
            obs_temp = []
            infos.append(info)
            observations.append(obs)
        return np.array(observations), np.array(infos)
    def convert_to_dict(self, action):
        obs_dict = {}
        for a in self.possible_agents:
            ind = self.possible_agents.index(a)
            obs_dict[a] = int(action[ind])
        return obs_dict

    def reset_env(self, ind):
        obs, info = self.envs[ind].reset()
        return obs, info
    
    def step(self, actions):
        """Step through the current environment. If done, switch to the next."""
        observations = []
        rewards = []
        truncations = []
        terminations = []
        infos = []
        reset_env = []
        final_infos = {}
        for e in range(self.num_envs):
            if e not in self.resets:
                obs, rew, trunc, term, info = self.envs[e].step(self.convert_to_dict(actions[e]))
                if trunc[self.possible_agents[0]] or term[self.possible_agents[0]]:
                    reset_env.append(e)
            else:
                obs, info = self.reset_env(e)
                term = {}
                trunc = {}
                rew = {}
                for a in self.possible_agents:
                    rew[a] = 0
                    term[a] = False
                    trunc[a] = False
            observations.append(obs)
            rewards.append(rew)
            terminations.append(term)
            truncations.append(trunc)
            infos.append(info)
        self.resets = reset_env

        return np.array(observations), np.array(rewards), np.array(terminations), np.array(truncations), np.array(infos)

    def render(self):
        """Render the current active environment."""
        return 
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


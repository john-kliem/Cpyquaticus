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
        #print(self.envs[0])
        self.agents = self.envs[0].agents  # Assuming all envs have the same agents
        self.possible_agents = self.envs[0].possible_agents

        # Assuming all environments have the same observation & action spaces
        self.observation_spaces = {agent: self.envs[0].observation_space(agent) for agent in self.possible_agents}
        #print("Observation_spaces: ", self.observation_spaces)
        self.action_spaces = {agent: self.envs[0].action_space(agent) for agent in self.possible_agents}
        self.num_steps = self.envs[0].max_steps
        self.resets = []
    def reset(self, seed=None, options=None):
        """Reset all environments and start from the first."""
        #self.env_states = [env.reset(seed=seed, options=options) for env in self.envs]
        observations = []
        infos = []
        for env in self.envs:
            obs, info = env.reset()
            obs_temp = []
            infos.append(info)
            for a in self.possible_agents:
                obs_temp.append(obs[a])
            observations.append(np.array(obs_temp))
        return np.array(observations), np.array(infos)
    def convert_to_dict(self, action):
        obs_dict = {}
        for a in self.possible_agents:
            ind = self.possible_agents.index(a)
            obs_dict[a] = int(action[ind])
        return obs_dict

    def reset_env(self, ind):
        obs, info = self.envs[ind].reset()
        #observations = []
        #obs_temp = []
        #infos = []
        #infos.append(info)
        #for a in self.possible_agents:
        #    obs_temp.append(obs[a])
            #info_temp.append(info[a])
        #observations.append(np.array(obs_temp))
        return obs, info#np.array(infos)
    
    def step(self, actions):
        """Step through the current environment. If done, switch to the next."""
        #obs, rewards, terminations, truncations, infos = self.envs[self.env_pos].step(actions)

        #if all(terminations.values()) or all(truncations.values()):
        #    self.env_pos += 1  # Move to the next environment
        #    if self.env_pos < self.num_envs:
        #        obs = self.envs[self.env_pos].reset()
        #    else:
        #        obs = {agent: np.zeros_like(self.observation_spaces[agent].sample()) for agent in self.agents}  # Dummy obs
        observations = []
        rewards = []
        truncations = []
        terminations = []
        infos = []
        reset_env = []
        #print("Actions: ", actions)
        for e in range(self.num_envs):
            temp_obs = []
            temp_rews = []
            temp_truncs = []
            temp_terms = []

            if e not in self.resets:
            #print("Action to convert: ", actions[e])
            #print("Converted Actions: ", self.convert_to_dict(actions[e]))
                obs, rew, trunc, term, info = self.envs[e].step(self.convert_to_dict(actions[e]))
                if term[self.possible_agents[-1]]:
                    info['final_info'] = [0,0]
                    reset_env.append(e)
                infos.append(info)
                for a in self.possible_agents:
                    temp_obs.append(obs[a])
                    temp_rews.append(rew[a])
                    temp_truncs.append(trunc[a])
                    temp_terms.append(term[a])
            else:
                obs, temp_infos = self.reset_env(e)
                for a in self.possible_agents:
                    temp_obs.append(obs[a])
                    temp_rews.append(0)
                    temp_truncs.append(False)
                    temp_terms.append(False)
                infos.append(temp_infos)
            observations.append(np.array(temp_obs))
            rewards.append(np.array(temp_rews))
            terminations.append(np.array(temp_terms))
            truncations.append(np.array(temp_truncs))
            #infos.append(temp_infos)
        self.resets = reset_env

        return np.array(observations), np.array(rewards), np.array(terminations), np.array(truncations), np.array(infos)

    def render(self):
        """Render the current active environment."""
        return #self.envs[self.env_pos].render()

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


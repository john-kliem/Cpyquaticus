import ctypes
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
from cpyquaticus.envs.c_binding import load_cpyquaticus
import copy
import pygame
import math
from cpyquaticus.envs.mctf_render import MCTFRender

#Allow visual of agents going out of bounds
FIELD_PADDING = 50

#Pyqyaticus Pettingzoo Environment With C Backend
class Cpyquaticus(ParallelEnv):
    metadata = {"render.modes": ["human",], "name": "cpyquaticus_v0"}

    def __init__(self, field_width=160, field_height=80, num_agents=2, num_teams=2, num_steps=1000, obs_type='real', normalized=True, render_mode=None, c_load = 'mac'):
        super().__init__()
        self.field_width = field_width
        self.field_height = field_height
        self.num_teams = num_teams
        self.num_steps = num_steps
        self.starting_size = num_agents
        self.render_mode = render_mode
        self.agents = ['agent_'+str(agent_id) for agent_id in range(self.starting_size)]
        self.possible_agents = self.agents[:]
        self.observation_spaces = {}
        self.action_spaces = {}
        self.normalized = normalized
        self.obs_length = None
        self.c_load = c_load
        
        if obs_type == 'real':
            self.obs_length = 8 * len(self.agents)
        elif obs_type == 'relative':
            self.obs_length = 16 * len(self.agents)
        self.game = None
        self.episode = None
        self.cpyquaticus = None
        self.steps = 0
        self.max_steps = 10 * self.num_steps # 4 minute fixed game
        self.cpyquaticus = load_cpyquaticus(c_load)
        self.observation_spaces = {agent_id: spaces.Box(-1,1, shape=(self.obs_length,),dtype=np.float32) for agent_id in self.agents}
        self.action_spaces = {agent_id: spaces.Discrete(18) for agent_id in self.agents}
        self.prev_agent_rewards = {agent_id:self.set_agent_rews(agent_id) for agent_id in self.agents}
        if self.render_mode == 'human':
            self._init_render()
    def set_agent_rews(self, agent_id):
        return {'agent':{'grab':0, 'capture':0, 'tag': 0, 'got_tagged':0, 'oob':0}, 'team':{'grab':0,'capture':0, 'opp_grab':0, 'opp_cap':0}}
    def get_rewards(self,):
        rewards = {}
        team_rews = {}
        agent_rews = {}
        #Unpack C Reward Structures
        for i in range(self.num_teams):
            team_rews[i] = self.game.contents.rewards[i].contents 
        for ind,agent_id in enumerate(self.agents):
            team = 1 if self.agents.index(agent_id) >= self.num_agents/2 else 0
            agent_rews = self.episode.contents.rewards[ind].contents
            rewards[agent_id] = 0.0; #-0.00040
            if agent_rews.oob > self.prev_agent_rewards[agent_id]['agent']['oob']:
                rewards[agent_id] += -1.0
            if agent_rews.grab > self.prev_agent_rewards[agent_id]['agent']['grab']:
                rewards[agent_id] += 0.25
            if agent_rews.capture > self.prev_agent_rewards[agent_id]['agent']['capture']:
                rewards[agent_id] += 1.0
            if team_rews[team].opp_grab > self.prev_agent_rewards[agent_id]['team']['opp_grab']:
                rewards[agent_id] += -0.25
            if team_rews[team].opp_capture > self.prev_agent_rewards[agent_id]['team']['opp_cap']:
                rewards[agent_id] += -1.0
            self.prev_agent_rewards[agent_id]['agent']['tag'] = agent_rews.tag
            self.prev_agent_rewards[agent_id]['agent']['grab'] = agent_rews.grab
            self.prev_agent_rewards[agent_id]['agent']['capture'] = agent_rews.capture
            self.prev_agent_rewards[agent_id]['agent']['got_tagged'] = agent_rews.got_tagged
            self.prev_agent_rewards[agent_id]['agent']['oob'] = agent_rews.oob
            #Update Prev Team Rewards
            self.prev_agent_rewards[agent_id]['team']['team_grab'] = team_rews[team].team_grab
            self.prev_agent_rewards[agent_id]['team']['team_cap'] = team_rews[team].team_capture
            self.prev_agent_rewards[agent_id]['team']['opp_grab'] = team_rews[team].opp_grab
            self.prev_agent_rewards[agent_id]['team']['opp_cap'] = team_rews[team].opp_capture
        return rewards
    def reset(self, seed=None, options=None):
        if not (self.game == None):
            self.close()
        self.game = self.cpyquaticus.create_game(self.field_width, self.field_height, self.starting_size, self.num_teams, self.max_steps, 1)
        self.episode = self.cpyquaticus.create_episode(self.game)
        self.steps = 0
        self.max_steps = 10 * self.num_steps#10 * 240 #4 Minute game
        self.cpyquaticus.initialize_game_starts(self.game, self.episode)
        self.agents = ['agent_'+str(agent_id) for agent_id in range(self.starting_size)]
        self.prev_agent_rewards = {agent_id:self.set_agent_rews(agent_id) for agent_id in self.agents}
        observations = self.get_agent_observations()
        return observations,{agent_id:{} for agent_id in self.agents}
    def step(self, actions):
        if self.game == None or self.episode == None:
            assert("")
        acts = [actions[k] for k in actions]
        c_actions = (ctypes.c_int * len(actions))(*acts)
        self.cpyquaticus.step(self.game, self.episode, c_actions)
        observations = self.get_agent_observations()
        self.steps += 1
        rewards = self.get_rewards()
        terminateds = {}
        truncateds = {}
        infos = {}
        remove = []
        for agent_id in self.agents:
            infos[agent_id] = {}
            if self.steps >= self.max_steps:
                terminateds[agent_id] = True
                truncateds[agent_id] = True 
                if agent_id in self.agents:
                    remove.append(agent_id)
            else:
                terminateds[agent_id] = False
                truncateds[agent_id] = False 
        for a in remove:
            self.agents.remove(a)
        if self.render_mode == 'human':
            self.render()
        return observations, rewards, terminateds, truncateds, infos
    def _init_render(self):
        self.renderer = MCTFRender()
    def render(self):
        self.renderer.objects = []
        for i in range(len(self.agents)):
            heading = self.episode.contents.agents[i].contents.heading
            pos = [self.episode.contents.agents[i].contents.pos[0],self.episode.contents.agents[i].contents.pos[1]]
            self.renderer.add_object(pos[0], pos[1], heading)
        self.renderer.run()
        return
    def close(self):
        self.cpyquaticus.free_game(self.game)
        self.game = None
        self.cpyquaticus.free_episode(self.episode, self.starting_size)
        self.episode = None
        if self.render_mode == 'human':
            self.renderer.close()
        return
    def get_agent_observations(self,):
        game_settings = self.game.contents
        if self.normalized:
            c_observations = self.cpyquaticus.get_observations(self.game, self.episode, 1, game_settings.max, game_settings.min)
        else:
            c_observations = self.cpyquaticus.get_observations(self.game, self.episode, 0, game_settings.max, game_settings.min)
        observations = {}
        index = 0
        for i in range(self.num_agents):  
            row_ptr = c_observations[i]  # Dereference the row pointer
            
            observations['agent_'+str(i)] = []
            for j in range(self.obs_length):  
                val = row_ptr[j]
                if val < -1:
                    val = -1
                elif val > 1:
                    val = 1
                observations['agent_'+str(i)].append(val)
        self.cpyquaticus.free_obs_list(c_observations, len(self.agents))
        return observations
    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]
    def action_space(self, agent_id):
        return self.action_spaces[agent_id]


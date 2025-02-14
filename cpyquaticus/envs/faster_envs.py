import ctypes
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
from c_binding import load_cpyquaticus
import copy
import pygame
import math
from mctf_render import MCTFRender
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
        # self.render = render
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
        # self.observation_space = self.observation_spaces['agent_0']
        # self.action_space = self.action_spaces['agent_0']

        # self.game = self.cpyquaticus.create_game(self.field_width, self.field_height, len(self.agents), self.num_teams, self.num_steps)
        # self.episode = self.cpyquaticus.create_episode(self.game)

        # # self.max = self.cpyquaticus.create_limit_obs(self.field_width+50.0, self.field_height+50.0, 180.0, 3.0, 60.0, 1);
        # # self.min = self.cpyquaticus.create_limit_obs(-50.0, -50.0, -180.0, 0.0, 0.0, 0);
        # self.steps = 0
        # self.cpyquaticus.initialize_game_starts(self.game, self.episode)
        
        # Initialize the game using the C function
        # self.game = self.cpyquaticus.create_game(field_width, field_height, num_agents, num_teams, num_steps)
        # if not self.game:
        #     raise ValueError("Failed to create game settings.")

        # self.episode = ctf_lib.create_episode(self.game)

        # self.agents = [f"agent_{i}" for i in range(num_agents)]
        # self.possible_agents = self.agents[:]
        # self.current_step = 0

        # # Action space: 17 discrete actions
        # self.action_space = spaces.Discrete(17)

        # # Observation space: [pos_x, pos_y, heading, speed, cooldown]
        # self.observation_space = spaces.Box(
        #     low=np.array([-50, -50, 0, 0, 0]),
        #     high=np.array([field_width + 50, field_height + 50, 360, 3.0, 60.0]),
        #     dtype=np.float32,
        # )
    def set_agent_rews(self, agent_id):
        return {'agent':{'grab':0, 'capture':0, 'tag': 0, 'got_tagged':0, 'oob':0}, 'team':{'grab':0,'capture':0, 'opp_grab':0, 'opp_cap':0}}
    def get_rewards(self,):
        rewards = {}
        team_rews = {}
        agent_rews = {}
        #Unpack C Reward Structures
        for i in range(self.num_teams):
            team_rews[i] = self.game.contents.rewards[i].contents #.team_rewards[i].contents.contents
        
        for ind,agent_id in enumerate(self.agents):
            
            team = 1 if self.agents.index(agent_id) >= self.num_agents/2 else 0
            agent_rews = self.episode.contents.rewards[ind].contents
            # rewards += team_rews[team]
            rewards[agent_id] = 0.0; #-0.00040
            
            # if agent_rews.got_tagged > self.prev_agent_rewards[agent_id]['agent']['got_tagged']:
            # #     # print("agent got tagged!!")
                # rewards[agent_id] += -0.1
            # # print("agent_id: ", agent_id, " OOB: ", agent_rews.oob, " PREV: ", self.prev_agent_rewards[agent_id]['agent']['oob'])
            if agent_rews.oob > self.prev_agent_rewards[agent_id]['agent']['oob']:
                # print("oob")
                rewards[agent_id] += -1.0
            if agent_rews.grab > self.prev_agent_rewards[agent_id]['agent']['grab']:
                # print("grab",end='')
                rewards[agent_id] += 0.25
            if agent_rews.capture > self.prev_agent_rewards[agent_id]['agent']['capture']:
                # print("capture",end='')
                rewards[agent_id] += 1.0
            if team_rews[team].opp_grab > self.prev_agent_rewards[agent_id]['team']['opp_grab']:
            #     # print("Opp Grab",end='')
                rewards[agent_id] += -0.25
            if team_rews[team].opp_capture > self.prev_agent_rewards[agent_id]['team']['opp_cap']:
            #     # print("Opp Capture",end='')
                rewards[agent_id] += -1.0
            # print()
            # print()
            #Update Prev rewards:
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
        observations = self.get_agent_observations()
        #print("Obs: ", observations)
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
                #print(row_ptr[j], end=", ")  # Dereference and print each float value
        # for i in range(num_agents):  # Loop through rows
        # row_ptr = matrix_ptr[i]  # Dereference row pointer
        # for j in range(num_features):  # Loop through columns
        #     print(f"Element [{i}][{j}] = {row_ptr[j]}")
        # for agent_id in self.agents:#
        #     # print("Obs: ", c_observations)
            
        #     observations[agent_id] = [c_observations[index].contents[i] for i in range(self.obs_length)]
        #     index += 1
        self.cpyquaticus.free_obs_list(c_observations, len(self.agents))

        # print("OBSERVATIONS: ", c_observations)
        return observations
    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]
    def action_space(self, agent_id):
        return self.action_spaces[agent_id]

    # def _get_observations(self):
    #     observations = []
    #     for i in range(self.num_agents):
    #         pos_x = self.episode.contents.agents[i].pos[0]
    #         pos_y = self.episode.contents.agents[i].pos[1]
    #         heading = self.episode.contents.agents[i].heading
    #         speed = self.episode.contents.agents[i].speed
    #         cooldown = self.episode.contents.agents[i].tagging_cooldown
    #         observations.append(np.array([pos_x, pos_y, heading, speed, cooldown], dtype=np.float32))
    #     return observations

    # def _get_rewards(self):
    #     return [self.episode.contents.rewards[i].capture for i in range(self.num_agents)]

    # def _check_done(self):
    #     done_flag = self.current_step >= self.num_steps
    #     return [done_flag] * self.num_agents

    # def render(self, mode="human"):
    #     if self.render:
    #         ctf_lib.render_game(self.game, self.episode)

    # def close(self):
    #     ctf_lib.free_episode(self.episode, self.num_agents)
    #     ctf_lib.free_game(self.game)
    #     if self.render_mode == 'human':
    #         self.renderer.close()
# if __name__ == "__main__":
#     vals = Cpyquaticus(render_mode='human')
#     obs, _ = vals.reset()
#     for i in range(2400):
#         obs, rew, _, _, _ = vals.step({'agent_0':vals.action_space('agent_0').sample(), 'agent_1':17})

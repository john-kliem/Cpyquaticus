import ctypes
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
from c_binding import load_cpyquaticus
import copy
#Pyqyaticus Pettingzoo Environment With C Backend
class Cpyquaticus(ParallelEnv):
    metadata = {"render.modes": ["human",], "name": "cpyquaticus_v0"}

    def __init__(self, field_width=160, field_height=80, num_agents=2, num_teams=2, num_steps=1000, obs_type='real', normalized=True, render=None):
        super().__init__()
        self.field_width = field_width
        self.field_height = field_height
        self.num_teams = num_teams
        self.num_steps = num_steps
        self.render = render
        self.agents = ['agent_'+str(agent_id) for agent_id in range(num_agents)]
        self.possible_agents = self.agents[:]
        self.observation_spaces = {}
        self.action_spaces = {}
        self.normalized = normalized
        self.obs_length = None
        if obs_type == 'real':
            self.obs_length = 8 * len(self.agents)
        elif obs_type == 'relative':
            self.obs_length = 16 * len(self.agents)
        self.game = None
        self.episode = None
        self.cpyquaticus = None
        self.steps = 0
        self.max_steps = 10 * 240 # 4 minute fixed game
        self.cpyquaticus = load_cpyquaticus()
        self.observation_spaces = {agent_id: spaces.Box(-1,1, shape=(self.obs_length,) ) for agent_id in self.agents}
        self.action_spaces = {agent_id: spaces.Discrete(17) for agent_id in self.agents}
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
    def get_rewards(self,):
        rewards = {}
        team_rews = {}
        agent_rews = {}
        #Unpack C Reward Structures
        for i in range(self.num_teams):
            team_rews[i] = self.game.contents.rewards[i].contents #.team_rewards[i].contents.contents
            agent_rews['agent_0'] = self.episode.contents.rewards[i].contents#.agent_rewards[i].contents.contents
        for agent_id in agent_rews:
            team = 1 if self.agents.index(agent_id) >= self.num_agents/2 else 0
            rewards[agent_id] = 0.0
            if agent_rews[agent_id].got_tagged:
                rewards[agent_id] += -1.0
            if agent_rews[agent_id].oob:
                rewards[agent_id] += -1.0
            if team_rews[team].team_grab:
                rewards[agent_id] += 0.5
            if team_rews[team].team_capture:
                rewards[agent_id] += 1.0
        return rewards
    def reset(self, seed=None, options=None):
        if not (self.game == None):
            self.close()
        self.game = self.cpyquaticus.create_game(self.field_width, self.field_height, len(self.agents), self.num_teams, self.max_steps, 1)
        self.episode = self.cpyquaticus.create_episode(self.game)
        self.steps = 0
        self.max_steps = 10 * 240 #4 Minute game
        self.cpyquaticus.initialize_game_starts(self.game, self.episode)
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
        for agent_id in self.agents:
            infos[agent_id] = {}
            if self.steps >= self.max_steps:
                terminateds[agent_id] = True
                truncateds[agent_id] = True 
            else:
                terminateds[agent_id] = False
                truncateds[agent_id] = False 
        return observations, rewards, terminateds, truncateds, infos
    def render(self):
        return
    def close(self):
        self.cpyquaticus.free_game(self.game)
        self.game = None
        self.cpyquaticus.free_episode(self.episode, len(self.agents))
        self.episode = None
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

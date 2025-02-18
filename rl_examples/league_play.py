class MCTFLeague:
    def __init__(self, num_main=1, num_exploiters=1):
        self.num_main = num_main
        self.num_exploiters = num_exploiters

        self.main_red = [None for i in range(num_main)]
        self.main_elos = [500 for i in range(num_main)]

        self.exploiters_red = [None for i in range(num_exploiters]
        self.exploiters_red_elos = [500 for i in range(num_exploiters)]

        self.prev_reds = []
        self.prev_reds_elos = []

        #Initialize Main Blue Agents
        self.main_blue = [None for i in range(num_main)]
        self.main_elos = [500 for i in range(num_main)]

        self.exploiters_blue = [None for i in range(num_exploiters)]
        self.exploiters_blue_elos = [500 for i in range(num_exploiters)]

        self.prev_blues = []
        self.prev_blues_elos = []


        self.selfplay_loops = 0
    def train(self, selfplay_loops=1):
        for loop in range(selfplay_loops):
            #Main Agents
            #Train Main Blue
            for mb in range(len(self.main_blue)):
                #Select Red Agents to train against if any
                b_algo = get_platform_algo(self.main_blue[mb], None)
                self.main_blue[mb] = self._train_blue(b_algo)
            #Train Main Red
            for mr in range(len(self.main_red)):
                #Select Red Agents to train against if any
                r_algo = get_platform_algo(self.main_red[mr], None)
                self.main_blue[mr] = self._train_red(r_algo)

            #Main Agent Exploiters
            #Train Blue Exploiter
            for eb in range(len(self.exploiters_blue)):
                #Select Red Agents to train against if any
                b_algo = get_platform_algo(None, self.main_agents)
                self.exploiters_red[eb] = self._train_red(b_algo)
                self.exploiters_blue_elos[eb] = 500
            #Train Red Exploiter
            for er in range(len(self.exploiters_red)):
                #Select Red Agents to train against if any
                r_algo = get_platform_algo(None, self.main_agents)
                self.exploiters_red[er] = self._train_red(r_algo)
                self.exploiters_red_elos[er] = 500

            #Evaluate and update elo scores for all agents
            for mb in range(len(self.main_blue)):
                #Eval Latest Red Main
                #Eval Blue Exploiter
                rew = self._run_blue_game(
        return
 


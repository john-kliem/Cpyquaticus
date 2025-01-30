#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef struct{
    //Key Game Events
    int team_grab;
    int team_capture;
    // Opponent Team Based Rewars
    int opp_grab;
    int opp_capture;

}team_rewards;

typedef struct{
    //Agent Rewards
    int tag;
    int grab;
    int capture;
    //Occurred to agent
    int got_tagged;
    int oob;

}agent_rewards;

team_rewards * create_team_reward(){
	team_rewards * reward = (team_rewards*)malloc(sizeof(team_rewards));
	reward->team_grab = 0;
	reward->team_capture = 0;
	reward->opp_grab = 0;
	reward->opp_capture = 0;
	return reward;
}

agent_rewards * create_agent_reward(){
	agent_rewards * reward = (agent_rewards*)malloc(sizeof(agent_rewards));
	reward->tag = 0;
	reward->grab = 0;
	reward->capture = 0;

	reward->got_tagged = 0;
	reward->oob = 0;

	return reward;
}
void free_team_reward(team_rewards* reward){
	free(reward);
}
void free_agent_reward(agent_rewards * reward){
	free(reward);
}


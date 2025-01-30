#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>



typedef struct{
    //Agent Rewards
    int tag;
    int grab;
    int capture;
    //Occurred to agent
    int got_tagged;
    int oob;
    //Team based Rewards
    int team_grab;
    int team_capture;
    // Opponent Team Based Rewars
    int opp_grab;
    int opp_capture;
}rewards;

rewards * create_reward(){
	rewards * reward = (rewards*)malloc(sizeof(rewards));
	reward->tag = 0;
	reward->grab = 0;
	reward->capture = 0;

	reward->got_tagged = 0;
	reward->oob = 0;

	reward->team_grab = 0;
	reward->team_capture = 0;
	
	reward->opp_grab = 0;
	reward->opp_capture = 0;

	return reward;
}

void free_reward(rewards * reward){
	free(reward);
}


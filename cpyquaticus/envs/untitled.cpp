#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <heron.h>

typedef struct{
	float bearing_to;// = 0.0;
	float distance_to;// = 0.0;
	float heading_to_you;// = 0.0;
	float speed;// = 0.0;
	int has_flag;// = 0;
	float on_their_side;// = 0.0;
	float tagging_cooldown;// = 0.0;
	int is_tagged;// = 0;
}oagent;

typedef struct {
	float opp_flag_bearing;// = 0.0;
	float opp_flag_distance;// = 0.0;
	float team_flag_bearing;// = 0.0;
	float team_flag_distance;// = 0.0;

	float wall_0_bearing;// = 0.0;
	float wall_0_distance;// = 0.0;
	float wall_1_bearing;// = 0.0;
	float wall_1_distance;// = 0.0;
	float wall_2_bearing;// = 0.0;
	float wall_2_distance;// = 0.0;
	float wall_3_bearing;// = 0.0;
	float wall_3_distance;// = 0.0;

	float scrim_bearing;// = 0.0;
	float scrim_distance;// = 0.0;

	float speed;// = 0.0;
	int own_flag;// = 1;
	int on_side;// = 1;
	float tag_cooldown;// = 0.0;
	oagent* other_agents[];
}OBS;

//list_obs
//Converts struct observation space into a list for a neural network
//Args: 
//normalize 1 puts all values between 1 and -1; 0 returns the true game values
//num_other_agents: number of other agents that are in the observation space
//Returns: list of floats (-1.0 - 1.0 if normalized)
float* list_obs(OBS* current_obs, int normalize, int num_other_agents, OBS* max, OBS* min){
	float * obs_list = malloc(sizeof(float) * 18 + 8 * num_other_agents * sizeof(float));
	obs_list[0] = 2.0;
	obs_list[1] = 2.0;
	obs_list[2] = 2.0;
	obs_list[3] = 2.0;
	obs_list[4] = 2.0;
}




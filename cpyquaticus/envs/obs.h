#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "heron.h"


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

float normalize(float value, float max, float min){
	return 2 * ((value - min) / (max - min)) - 1;
}
//list_obs
//Converts struct observation space into a list for a neural network
//Args: 
//norm 1 puts all values between 1 and -1; 0 returns the true game values
//num_other_agents: number of other agents that are in the observation space
//Returns: list of floats (-1.0 - 1.0 if normalized)
float* norm_list_obs(OBS* current_obs, int num_other_agents, OBS* max, OBS* min){
	float * obs_list = malloc(sizeof(float) * 18 + 8 * num_other_agents * sizeof(float));
	obs_list[0] = normalize(current_obs->opp_flag_bearing, max->opp_flag_bearing, min->opp_flag_bearing);
	obs_list[1] = normalize(current_obs->opp_flag_distance, max->opp_flag_distance, min->opp_flag_distance);
	obs_list[2] = normalize(current_obs->team_flag_bearing, max->team_flag_bearing, min->team_flag_bearing);
	obs_list[3] = normalize(current_obs->team_flag_distance, max->team_flag_distance, min->team_flag_distance);
	obs_list[4] = normalize(current_obs->wall_0_bearing, max->wall_0_bearing, min->wall_0_bearing);
	obs_list[5] = normalize(current_obs->wall_0_distance, max->wall_0_distance, min->wall_0_distance);
	obs_list[6] = normalize(current_obs->wall_1_bearing, max->wall_1_bearing, min->wall_1_bearing);
	obs_list[7] = normalize(current_obs->wall_1_distance, max->wall_1_distance, min->wall_1_distance);
	obs_list[8] = normalize(current_obs->wall_2_bearing, max->wall_2_bearing, min->wall_2_bearing);
	obs_list[9] = normalize(current_obs->wall_2_distance, max->wall_2_distance, min->wall_2_distance);
	obs_list[10] = normalize(current_obs->wall_3_bearing, max->wall_3_bearing, min->wall_3_bearing);
	obs_list[11] = normalize(current_obs->wall_3_distance, max->wall_3_distance, min->wall_3_distance);
	obs_list[12] = normalize(current_obs->scrim_bearing, max->scrim_bearing, min->scrim_bearing);
	obs_list[13] = normalize(current_obs->scrim_distance, max->scrim_distance, min->scrim_distance);
	obs_list[14] = normalize(current_obs->speed, max->speed, min->speed);
	obs_list[15] = normalize(current_obs->own_flag, max->own_flag, min->own_flag);
	obs_list[16] = normalize(current_obs->on_side, max->on_side, min->on_side);
	obs_list[17] = normalize(current_obs->tag_cooldown, max->tag_cooldown, min->tag_cooldown);
	for (int i = 0; i < num_other_agents; i++){
		obs_list[18+i] = normalize(current_obs->other_agents[i]->bearing_to, max->other_agents[i]->bearing_to, min->other_agents[i]->bearing_to);
		obs_list[19+i] = normalize(current_obs->other_agents[i]->distance_to, max->other_agents[i]->distance_to, min->other_agents[i]->distance_to);
		obs_list[20+i] = normalize(current_obs->other_agents[i]->heading_to_you, max->other_agents[i]->heading_to_you, min->other_agents[i]->heading_to_you);
		obs_list[21+i] = normalize(current_obs->other_agents[i]->speed, max->other_agents[i]->speed, min->other_agents[i]->speed);
		obs_list[22+i] = normalize(current_obs->other_agents[i]->has_flag, max->other_agents[i]->has_flag, min->other_agents[i]->has_flag);
		obs_list[23+i] = normalize(current_obs->other_agents[i]->on_their_side, max->other_agents[i]->on_their_side, min->other_agents[i]->on_their_side);
		obs_list[24+i] = normalize(current_obs->other_agents[i]->tagging_cooldown, max->other_agents[i]->tagging_cooldown, min->other_agents[i]->tagging_cooldown);
		obs_list[25+i] = normalize(current_obs->other_agents[i]->is_tagged, max->other_agents[i]->is_tagged, min->other_agents[i]->is_tagged);
	}
	return obs_list;
}

float* list_obs(OBS* current_obs, int num_other_agents, OBS* max, OBS* min){
	float *obs_list = malloc(sizeof(float) * 18 + 8 * num_other_agents * sizeof(float));
	obs_list[0] = current_obs->opp_flag_bearing;
	obs_list[1] = current_obs->opp_flag_distance;
	obs_list[2] = current_obs->team_flag_bearing;
	obs_list[3] = current_obs->team_flag_distance;
	obs_list[4] = current_obs->wall_0_bearing;
	obs_list[5] = current_obs->wall_0_distance;
	obs_list[6] = current_obs->wall_1_bearing;
	obs_list[7] = current_obs->wall_1_distance;
	obs_list[8] = current_obs->wall_2_bearing;
	obs_list[9] = current_obs->wall_2_distance;
	obs_list[10] = current_obs->wall_3_bearing;
	obs_list[11] = current_obs->wall_3_distance;
	obs_list[12] = current_obs->scrim_bearing;
	obs_list[13] = current_obs->scrim_distance;
	obs_list[14] = current_obs->speed;
	obs_list[15] = current_obs->own_flag;
	obs_list[16] = current_obs->on_side;
	obs_list[17] = current_obs->tag_cooldown;
	for (int i = 0; i < num_other_agents; i++) {
	    obs_list[18 + i] = current_obs->other_agents[i]->bearing_to;
	    obs_list[19 + i] = current_obs->other_agents[i]->distance_to;
	    obs_list[20 + i] = current_obs->other_agents[i]->heading_to_you;
	    obs_list[21 + i] = current_obs->other_agents[i]->speed;
	    obs_list[22 + i] = current_obs->other_agents[i]->has_flag;
	    obs_list[23 + i] = current_obs->other_agents[i]->on_their_side;
	    obs_list[24 + i] = current_obs->other_agents[i]->tagging_cooldown;
	    obs_list[25 + i] = current_obs->other_agents[i]->is_tagged;
	}
	return obs_list;
}



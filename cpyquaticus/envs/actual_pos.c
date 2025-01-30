#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "heron.h"


typedef struct{
	float x;// = 0.0;
	float y;// = 0.0;
	float heading;// = 0.0;
	float speed;// = 0.0;
	int has_flag;// = 0;
	int on_their_side;// = 0.0;
	float tagging_cooldown;// = 0.0;
	int is_tagged;// = 0;
}real_obs;

float normalize(float value, float max, float min){
	return 2 * ((value - min) / (max - min)) - 1;
}

real_obs* create_limit_obs(float x, float y, float heading, float speed, float tagging_cooldown, int is_max){
	real_obs * obs = malloc(sizeof(real_obs) * 8);
	obs->x = x;
	obs->y = y;
	obs->heading = heading;
	obs->speed = speed;
	obs->tagging_cooldown = tagging_cooldown;
	if (is_max){
		obs->has_flag = 1;
		obs->on_their_side = 1;
		obs->is_tagged = 1;
	}
	else{
		obs->has_flag = 0;
		obs->on_their_side = 0;
		obs->is_tagged = 0;
	}
	return obs;

}
real_obs* generate_obs(int agent_id, USV* agent)
{
	real_obs * obs = malloc(sizeof(real_obs) * 8);
	obs->x = agent->pos[0];
	obs->y = agent->pos[1];
	obs->heading = agent->heading;
	obs->speed = agent->speed;
	obs->has_flag = agent->has_flag;
	obs->on_their_side = agent->on_their_side;
	obs->tagging_cooldown = agent->tagging_cooldown;
	obs->is_tagged = agent->is_tagged;
	return obs;
}


//list_obs
//Converts struct observation space into a list for a neural network
//Args: 
//norm 1 puts all values between 1 and -1; 0 returns the true game values
//num_other_agents: number of other agents that are in the observation space
//Returns: list of floats (-1.0 - 1.0 if normalized)
float* norm_list_obs(int agent_id, real_obs * current_obs ,int num_agents, real_obs* max, real_obs* min){
	float * obs_list = malloc(sizeof(float) * 8 + 8 * (num_agents-1) * sizeof(float));
	obs_list[0] = normalize(current_obs[agent_id].x, max->x, min->x);
	obs_list[1] = normalize(current_obs[agent_id].y, max->y, min->y);
	obs_list[2] = normalize(current_obs[agent_id].heading, max->heading, min->heading);
	obs_list[3] = normalize(current_obs[agent_id].speed, max->speed, min->speed);
	obs_list[4] = normalize(current_obs[agent_id].has_flag, max->has_flag, min->has_flag);
	obs_list[5] = normalize(current_obs[agent_id].on_their_side, max->on_their_side, min->on_their_side);
	obs_list[6] = normalize(current_obs[agent_id].tagging_cooldown, max->tagging_cooldown, min->tagging_cooldown);
	obs_list[7] = normalize(current_obs[agent_id].is_tagged, max->is_tagged, min->is_tagged);
	for (int i = 0; i < num_agents; i++){
		if (i == agent_id)
		{
			continue;
		}
		else{
			obs_list[8+i] = normalize(current_obs[i].x, max->x, min->x);
			obs_list[9+i] = normalize(current_obs[i].y, max->y, min->y);
			obs_list[10+i] = normalize(current_obs[i].heading, max->heading, min->heading);
			obs_list[11+i] = normalize(current_obs[i].speed, max->speed, min->speed);
			obs_list[12+i] = normalize(current_obs[i].has_flag, max->has_flag, min->has_flag);
			obs_list[13+i] = normalize(current_obs[i].on_their_side, max->on_their_side, min->on_their_side);
			obs_list[14+i] = normalize(current_obs[i].tagging_cooldown, max->tagging_cooldown, min->tagging_cooldown);
			obs_list[15+i] = normalize(current_obs[i].is_tagged, max->is_tagged, min->is_tagged);
		}
	}
	return obs_list;
}

float* list_obs(real_obs* current_obs, int agent_id, int num_agents){
	float * obs_list = malloc(sizeof(float) * 8 + 8 * (num_agents-1) * sizeof(float));
	obs_list[0] = current_obs[agent_id].x;
	obs_list[1] = current_obs[agent_id].y;
	obs_list[2] = current_obs[agent_id].heading;
	obs_list[3] = current_obs[agent_id].speed;
	obs_list[4] = current_obs[agent_id].has_flag;
	obs_list[5] = current_obs[agent_id].on_their_side;
	obs_list[6] = current_obs[agent_id].tagging_cooldown;
	obs_list[7] = current_obs[agent_id].is_tagged;
	for (int i = 0; i < num_agents; i++){
		if (i == agent_id){
			continue;
		}
		else{
			obs_list[8+i] = current_obs[i].x;
			obs_list[9+i] = current_obs[i].y;
			obs_list[10+i] = current_obs[i].heading;
			obs_list[11+i] = current_obs[i].speed;
			obs_list[12+i] = current_obs[i].has_flag;
			obs_list[13+i] = current_obs[i].on_their_side;
			obs_list[14+i] = current_obs[i].tagging_cooldown;
			obs_list[15+i] = current_obs[i].is_tagged;
		}
	}
	return obs_list;
}

void free_obs(real_obs * obs){
	free(obs);
}

void free_obs_list(real_obs** obs_list, int num_agents) {
    if (obs_list != NULL) {
        for (int i = 0; i < num_agents; i++) {
            if (obs_list[i] != NULL) {
                free(obs_list[i]);  // Free each allocated real_obs struct
            }
        }
        free(obs_list);  // Free the main list holding the pointers
    }
}






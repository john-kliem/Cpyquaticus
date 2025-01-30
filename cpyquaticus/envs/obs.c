#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "heron.h"

typedef struct{
    float bearing_to;
    float distance_to;
    float heading_to_you;
    float speed;
    int has_flag;
    float on_their_side;
    float tagging_cooldown;
    int is_tagged;
} oagent;

typedef struct {
    float opp_flag_bearing;
    float opp_flag_distance;
    float team_flag_bearing;
    float team_flag_distance;
    float wall_0_bearing;
    float wall_0_distance;
    float wall_1_bearing;
    float wall_1_distance;
    float wall_2_bearing;
    float wall_2_distance;
    float wall_3_bearing;
    float wall_3_distance;
    float scrim_bearing;
    float scrim_distance;
    float speed;
    int own_flag;
    int on_side;
    float tag_cooldown;
    oagent* other_agents; // array of agents
} OBS;

OBS* define_min(float min_bearing, float min_distance, float min_tag_cooldown, float min_speed, int num_others) {
    OBS* min_obs = malloc(sizeof(OBS));
    min_obs->opp_flag_bearing = min_bearing;
    min_obs->opp_flag_distance = min_distance;
    min_obs->team_flag_bearing = min_bearing;
    min_obs->team_flag_distance = min_distance;
    min_obs->wall_0_bearing = min_bearing;
    min_obs->wall_0_distance = min_distance;
    min_obs->wall_1_bearing = min_bearing;
    min_obs->wall_1_distance = min_distance;
    min_obs->wall_2_bearing = min_bearing;
    min_obs->wall_2_distance = min_distance;
    min_obs->wall_3_bearing = min_bearing;
    min_obs->wall_3_distance = min_distance;
    min_obs->scrim_bearing = min_bearing;
    min_obs->scrim_distance = min_distance;
    min_obs->speed = min_speed;
    min_obs->own_flag = 0;
    min_obs->on_side = 0;
    min_obs->tag_cooldown = 0.0;

    // Allocate memory for other_agents (num_others agents)
    min_obs->other_agents = (oagent*) malloc(sizeof(oagent) * num_others);
    for (int i = 0; i < num_others; i++) {
        min_obs->other_agents[i].bearing_to = min_bearing;
        min_obs->other_agents[i].distance_to = 0.0;
        min_obs->other_agents[i].heading_to_you = min_bearing;
        min_obs->other_agents[i].speed = 0.0;
        min_obs->other_agents[i].has_flag = 0;
        min_obs->other_agents[i].on_their_side = 0;
        min_obs->other_agents[i].tagging_cooldown = 0.0;
        min_obs->other_agents[i].is_tagged = 0;
    }

    return min_obs;
}

OBS* define_max(float max_bearing, float max_distance, float max_tag_cooldown, float max_speed, int num_others) {
    OBS* max_obs = malloc(sizeof(OBS));
    max_obs->opp_flag_bearing = max_bearing;
    max_obs->opp_flag_distance = max_distance;
    max_obs->team_flag_bearing = max_bearing;
    max_obs->team_flag_distance = max_distance;
    max_obs->wall_0_bearing = max_bearing;
    max_obs->wall_0_distance = max_distance;
    max_obs->wall_1_bearing = max_bearing;
    max_obs->wall_1_distance = max_distance;
    max_obs->wall_2_bearing = max_bearing;
    max_obs->wall_2_distance = max_distance;
    max_obs->wall_3_bearing = max_bearing;
    max_obs->wall_3_distance = max_distance;
    max_obs->scrim_bearing = max_bearing;
    max_obs->scrim_distance = max_distance;
    max_obs->speed = max_speed;
    max_obs->own_flag = 1;
    max_obs->on_side = 1;
    max_obs->tag_cooldown = max_tag_cooldown;

    // Allocate memory for other_agents (num_others agents)
    max_obs->other_agents = (oagent*) malloc(sizeof(oagent) * num_others);
    for (int i = 0; i < num_others; i++) {
        max_obs->other_agents[i].bearing_to = max_bearing;
        max_obs->other_agents[i].distance_to = max_distance;
        max_obs->other_agents[i].heading_to_you = max_bearing;
        max_obs->other_agents[i].speed = max_speed;
        max_obs->other_agents[i].has_flag = 1;
        max_obs->other_agents[i].on_their_side = 1;
        max_obs->other_agents[i].tagging_cooldown = max_tag_cooldown;
        max_obs->other_agents[i].is_tagged = 1;
    }

    return max_obs;
}

OBS* init_obs(int num_others) {
    OBS* init_obs = malloc(sizeof(OBS));
    init_obs->opp_flag_bearing = 0.0;
    init_obs->opp_flag_distance = 0.0;
    init_obs->team_flag_bearing = 0.0;
    init_obs->team_flag_distance = 0.0;
    init_obs->wall_0_bearing = 0.0;
    init_obs->wall_0_distance = 0.0;
    init_obs->wall_1_bearing = 0.0;
    init_obs->wall_1_distance = 0.0;
    init_obs->wall_2_bearing = 0.0;
    init_obs->wall_2_distance = 0.0;
    init_obs->wall_3_bearing = 0.0;
    init_obs->wall_3_distance = 0.0;
    init_obs->scrim_bearing = 0.0;
    init_obs->scrim_distance = 0.0;
    init_obs->speed = 0.0;
    init_obs->own_flag = 0;
    init_obs->on_side = 1;
    init_obs->tag_cooldown = 0.0;

    // Allocate memory for other_agents (num_others agents)
    init_obs->other_agents = (oagent*) malloc(sizeof(oagent) * num_others);
    for (int i = 0; i < num_others; i++) {
        init_obs->other_agents[i].bearing_to = 0.0;
        init_obs->other_agents[i].distance_to = 0.0;
        init_obs->other_agents[i].heading_to_you = 0.0;
        init_obs->other_agents[i].speed = 0.0;
        init_obs->other_agents[i].has_flag = 0;
        init_obs->other_agents[i].on_their_side = 0;
        init_obs->other_agents[i].tagging_cooldown = 0.0;
        init_obs->other_agents[i].is_tagged = 0;
    }

    return init_obs;
}

void free_obs(OBS* observation, int num_others) {
    free(observation->other_agents);
    free(observation);
}

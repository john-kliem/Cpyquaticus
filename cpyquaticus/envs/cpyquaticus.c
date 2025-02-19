#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "heron.h"
#include "actual_pos.h"
#include "flag.h"
// #include "render_test.h"
#include "rewards.h"
#include <unistd.h>
#include <time.h>
#define ACTION_MAP_SIZE 17
#define NUM_HEADINGS 8
#define NUM_SPEEDS 2
const float HEADINGS[NUM_HEADINGS] = {0, 45, 90, 135, 180, 225, 270, 315};//{180, 135, 90, 45, 0, -45, -90, -135};
const float SPEEDS[NUM_SPEEDS] = {3.0, 1.5};
typedef struct{
    float speed;
    float heading;
} action;

typedef struct{
    //game settings
    int field_width;
    int field_height;
    int num_agents;
    int num_teams;
    int num_steps;
    int speed_up_factor;
    float grab_distance;
    float tagging_distance;
    float tagging_cooldown_max;
    action*  discrete_action_space;
    team_rewards** rewards;
    //Max Obs
    real_obs * max;
    //Min Obs
    real_obs * min;
    flag ** flags;
    //Include linked list of episodes or dynamic array?
}settings;

settings * create_game(int field_width, int field_height, int num_agents, int num_teams, int num_steps, int speed_up_factor){
    if(num_agents%num_teams != 0)
    {
        printf("ERROR: Number of agents needs to be equally divisible by the number of teams!");
        return NULL;
    }
    settings * game = (settings*)malloc(sizeof(settings));
    game->field_width = field_width;
    game->field_height = field_height;
    game->speed_up_factor = speed_up_factor;
    game->tagging_distance = 10.0;
    game->grab_distance = 10.0;
    game->num_agents = num_agents;
    game->num_teams = num_teams;
    game->num_steps = num_steps;
    game->tagging_cooldown_max = 60.0;
    game->max = create_limit_obs(field_width+50, field_height+50, 180.0, 3.0, 60.0, 1);
    game->min = create_limit_obs(-50, -50, -180.0, 0.0, 0.0, 0);
    game->discrete_action_space = (action*)malloc(ACTION_MAP_SIZE * sizeof(action));
    game->flags = (flag**)malloc(game->num_teams * sizeof(flag*));
    game->rewards = (team_rewards**)malloc(game->num_teams * sizeof(team_rewards*));
    for (int i = 0; i < game->num_teams; i++){
        game->flags[i] = create_flag(i, 0.0, 0.0);
        game->rewards[i] = create_team_reward();
    }
    int index = 0;
    for (int i = 0; i < NUM_SPEEDS; i++){
        for (int j = 0; j < NUM_HEADINGS; j++){
            game->discrete_action_space[index].speed = SPEEDS[i];
            game->discrete_action_space[index].heading = HEADINGS[j];
            index++;
        }
        //index = 0;
    }
    game->discrete_action_space[index].speed = 0.0;
    game->discrete_action_space[index].heading = 0.0;

    return game;
}

void free_game(settings * game){
    free_obs(game->max);
    free_obs(game->min);
    free(game->discrete_action_space);
    for(int i = 0; i < game->num_teams; i++){
        free_flag(game->flags[i]);
        free_team_reward(game->rewards[i]);
    }
    free(game->rewards);
    free(game->flags);
    free(game);
    return;
}

typedef struct {
    USV** agents;
    agent_rewards** rewards;
    // real_obs ** observations;
}episode;


episode * create_episode(settings* game){
    episode * e = (episode*)malloc(sizeof(episode));
    e->agents = (USV**)malloc(game->num_agents * sizeof(USV*));
    e->rewards = (agent_rewards**)malloc(game->num_agents*sizeof(agent_rewards*));
    // e->observations = (real_obs**)malloc(game->num_agents * sizeof(real_obs));
    if (e->agents == NULL){
        printf("Failed to allocate agent list");
        return NULL;
    }
    for (int i = 0; i < game->num_agents; i++){
        if(i < game->num_agents/2){
            e->agents[i] = create_heron(i, 0);
        }
        else{
            e->agents[i] = create_heron(i, 1);
        }
        e->rewards[i] = create_agent_reward();
    }

    return e;
}
void free_episode(episode* e, int num_agents){
    for (int i = 0; i < num_agents; i++){
        free_heron(e->agents[i]);
        free_agent_reward(e->rewards[i]);
    }
    free(e->rewards);
    free(e->agents);
    // free(e->observations);
    free(e);
}

//Computes the distance between two points
float distance(float x1, float y1, float x2, float y2){
    float distance = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
    return distance;
}
//Checks tags, grabs, and captures - limit # of for loops
void check_game_events(settings*game, episode* e){
    int oob = 0;
    for(int agent_id = 0; agent_id < game->num_agents; agent_id++){
        e->agents[agent_id]->tagging_cooldown = fmax((e->agents[agent_id]->tagging_cooldown - 0.1f),0.0f);
        // printf("Agent ID: %d, Tagging Cooldown: %f\n", agent_id, e->agents[agent_id]->tagging_cooldown);
        //Check side
        //Enforce left team_id 0 Right team_id 1 FIX for dynamic team numbers later
        if(e->agents[agent_id]->pos[0] > game->field_width || e->agents[agent_id]->pos[0] < 0){
            oob = 1;
        }
        if (e->agents[agent_id]->pos[1] > game->field_height || e->agents[agent_id]->pos[1] < 0){
            oob = 1;
        }

        if(oob && !e->agents[agent_id]->is_tagged){
            e->agents[agent_id]->is_tagged = 1;
            e->rewards[agent_id]->oob++;
            // printf("AGENT %d WENT OOB NEW SCORE: %d\n",agent_id, e->rewards[agent_id]->oob);
            if (e->agents[agent_id]->has_flag){
                // game->flags[e->agents[agent_id]->team]->grabbed_by = -1;
                // game->flags[e->agents[agent_id]->team]->grabbed = 0;
                e->agents[agent_id]->has_flag = 0;
                for(int i = 0; i < game->num_teams; i++){
                    if (game->flags[i]->grabbed_by == agent_id){
                        game->flags[i]->grabbed_by = -1;
                        game->flags[i]->grabbed = 0;
                    }
                }
            }
        }
        oob = 0;
        // else {
        //     if (e->rewards[agent_id]->oob > 0 && e->agents[agent_id]->is_tagged){
        //         printf("RETURNING AGENT: %d to 0\n", agent_id);
        //         e->rewards[agent_id]->oob = 0;
        //     }
        // }
        if(e->agents[agent_id]->team){
            if(e->agents[agent_id]->pos[0] > game->field_width/2){
                e->agents[agent_id]->on_their_side = 1;
            }
            else{
                e->agents[agent_id]->on_their_side = 0;
            }
        }
        else{
            if(e->agents[agent_id]->pos[0] < game->field_width/2){
                e->agents[agent_id]->on_their_side = 1;
            }
            else{
                e->agents[agent_id]->on_their_side = 0;
            }
        }
        
        //Check Player to see if should be untagged
        float flag_dist = distance(e->agents[agent_id]->pos[0], e->agents[agent_id]->pos[1],game->flags[e->agents[agent_id]->team]->pos[0],game->flags[e->agents[agent_id]->team]->pos[1]);
        if (flag_dist <= game->grab_distance && e->agents[agent_id]->is_tagged){
            // printf("Untaggng: %d\n", agent_id);
            e->agents[agent_id]->is_tagged = 0;
        }

        //Check Captures
        if (flag_dist <= game->grab_distance && e->agents[agent_id]->has_flag){
            e->agents[agent_id]->has_flag = 0;
            
            e->rewards[agent_id]->capture++;
            // printf("Agent: %f Capture\n", agent_id);
            for(int flag_id = 0; flag_id < game->num_teams; flag_id++){
                if(game->flags[flag_id]->grabbed_by == agent_id){
                    game->rewards[flag_id]->opp_capture++;
                    game->flags[flag_id]->grabbed_by = -1;
                    game->flags[flag_id]->grabbed = 0;
                }
                
            }
        }
        
        
        //Check for tags
        for (int other_id = 0; other_id < game->num_agents; other_id++){
            
            //TODO: This if is hideous fix at some point
            if(e->agents[agent_id]->tagging_cooldown <= 0.0 && e->agents[agent_id]->on_their_side && agent_id != other_id && e->agents[other_id]->team != e->agents[agent_id]->team && !e->agents[agent_id]->is_tagged && !e->agents[other_id]->on_their_side && !e->agents[other_id]->is_tagged){
                
                float dist = distance(e->agents[agent_id]->pos[0],e->agents[agent_id]->pos[1],e->agents[other_id]->pos[0],e->agents[other_id]->pos[1]);
                if (dist <= game->tagging_distance){
                    e->agents[other_id]->is_tagged = 1;
                    e->rewards[other_id]->got_tagged++;
                    e->rewards[agent_id]->tag++;
                    e->agents[agent_id]->tagging_cooldown = game->tagging_cooldown_max;
                    // printf("Agent %f Tagged\n", other_id); 
                    if (e->agents[other_id]->has_flag){
                        for(int team_id = 0; team_id < game->num_teams; team_id++){
                            if(game->flags[team_id]->grabbed_by == other_id){
                                game->flags[e->agents[agent_id]->team]->grabbed_by = -1;
                                game->flags[e->agents[agent_id]->team]->grabbed = 0;
                                
                            }
                        }
                        e->agents[other_id]->has_flag = 0;   
                    }
                }
        
            }
            
        }
        //Check to see if any grabs have occured
        if(!e->agents[agent_id]->is_tagged){
            for(int team_id = 0; team_id < game->num_teams; team_id++){
                if (game->flags[team_id]->team != e->agents[agent_id]->team && !game->flags[team_id]->grabbed){
                    flag_dist = distance(e->agents[agent_id]->pos[0], e->agents[agent_id]->pos[1],game->flags[team_id]->pos[0],game->flags[team_id]->pos[1]);
                    if (flag_dist <= game->grab_distance){
                        // printf("Grab\n");
                        e->agents[agent_id]->has_flag = 1;
                        e->rewards[agent_id]->grab++;
                        game->rewards[e->agents[agent_id]->team]->team_grab++;
                        game->rewards[team_id]->opp_grab++;
                        game->flags[team_id]->grabbed_by = agent_id;
                        game->flags[team_id]->grabbed = 1;
                    }
                }
                
            }
        }


    }
    return;
}


// void check_grabs(settings*game, episode* e){
//     for(int agent_id = 0; agent_id < game->num_agents; agent_id++){
//         //Check Player to see if should be untagged
        
//     return;
// }
//Only Handles Discrete Actions
float to_360(float angle) {
    return (angle < 0) ? angle + 360.0f : angle;
}
void step(settings* game, episode * e, int * actions){
    for(int sec = 0; sec < game->speed_up_factor; sec++){
        for(int i = 0; i < game->num_agents; i++){
            // printf("Picked Heading: %f cur heading: %f\n", game->discrete_action_space[actions[i]].heading, e->agents[i]->heading);
            float heading =  angle180(game->discrete_action_space[actions[i]].heading) - angle180(e->agents[i]->heading);
            float speed = game->discrete_action_space[actions[i]].speed;
            if(e->agents[i]->is_tagged){
                // printf("Agent: %d is Tagged Has Flag?: %d\n", i, e->agents[i]->has_flag);
                float cur_hdg = e->agents[i]->heading;
                float x  = e->agents[i]->pos[0];
                float y  = e->agents[i]->pos[1];
                float x2 = game->flags[e->agents[i]->team]->pos[0];
                float y2 = game->flags[e->agents[i]->team]->pos[1];
                float diff_y = y2-y;
                float diff_x = x2-x;
                float smallest_angle = (atan2((y2-y) ,(x2-x))*180.0/M_PI);
                float atanval = atan((y2-y)/ (x2-x))*180.0f/M_PI;
                float angle_diff;
                angle_diff = fmod(450-to_360(smallest_angle),360) - to_360(cur_hdg);
                float angle_diff2 = fmod(450-to_360(smallest_angle),360) - 360+ to_360(cur_hdg);
                // if (fabs(angle_diff) > fabs(angle_diff2)){
                //     angle_diff = angle_diff2;
                // }
                float heading_diff;

                if (fabs(angle_diff) < 10.0f ){
                    heading_diff = angle_diff;
                }
                else{

                    heading_diff = (angle_diff < 0) ? -175.0f : 175.0f;
                }
                if (i==1){
                move_heron(e->agents[i], 0.5, angle180(heading_diff));
                }
                else{
                move_heron(e->agents[i], 0.5, angle180(heading_diff)); 
                }
            }
            else{
                if(i==1){
                    move_heron(e->agents[i], speed, angle180(heading));
                }
                else{
                    move_heron(e->agents[i], speed, angle180(heading));
                }
            }
            check_game_events(game, e);
        }
        // sleep(1);
    }
}
// This isn't needed here TODO think about removing
episode * reset(settings*game){
    return create_episode(game);
}

float get_random_value(int min, int max){
    return (float)(min + rand() % (max-min+1));
}
void initialize_game_starts(settings * game, episode * e){
    int width = game->field_width/2.0; // Get Team Field Size
    int height = game->field_height; // Get Team Field Height
    //Set Flags 15m from start Pos
    for(int i = 0; i < game->num_teams; i++)
    {   
        //Flag Homes will be 15m off backline in the middile of the field height
        if (i == 0){
            game->flags[i]->pos[0] = 15;
            game->flags[i]->pos[1] = height/2;
        }
        else{
            game->flags[i]->pos[0] = game->field_width-15;
            game->flags[i]->pos[1] = height/2;
        }
    }
    //Assign Player Starting Positions
    //TODO: Check to make sure player isn't put on team flag spawn
    int i2 = game->num_agents/2;
    for(int i = 0; i < game->num_agents/2; i++){
        float x = get_random_value(10, width-10);
        float y = get_random_value(10, height-10);
        float hdg = get_random_value(0,359);//angle180(0.0f);//get_random_value(0,359));
            e->agents[i]->pos[0] = x;
            e->agents[i]->pos[1] = y;
            e->agents[i]->heading = hdg;
            e->agents[i2]->pos[0] = game->field_width - x;
            e->agents[i2]->pos[1] = y;
            e->agents[i2]->heading = hdg;
    }
}
//
float ** get_observations(settings*game, episode * e, int normalize, real_obs * max, real_obs * min){
    real_obs ** obs = (real_obs**)malloc(game->num_agents* sizeof(real_obs*));
    float ** final_observations = (float**)malloc(game->num_agents*sizeof(float*));
    for(int i = 0; i < game->num_agents; i++){
        obs[i] = generate_obs(i, e->agents[i]);
    }
    for(int i = 0; i < game->num_agents; i++){
        if (normalize){
            final_observations[i] = norm_list_obs(i, obs,game->num_agents, max, min);
        }  
        else{
            final_observations[i] = list_obs(obs, i, game->num_agents);
        }
    }
    free_obs_struct(obs, game->num_agents);
    // for (int i = 0; i < game->num_agents; i++){
    //     printf("CAgent ID: %d: ",i);
    //     for (int x = 0; x < game->num_agents*8; x++){
    //         printf("%f, ", final_observations[i][x]);
    //     }
    //     printf("\n");
    // }
    return final_observations;
}

// Compile for Python:
// gcc -shared -o cpyquaticus.dylib cpyquaticus.c -fPIC
// Render Compile for Python

// Mac
// clang -shared -fPIC -dynamiclib cpyquaticus.c -L lib/ -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL lib/libraylib.a -o cpyquaticus.dylib

// Linux
// clang -shared -fPIC cpyquaticus.c -L lib/ -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL lib/libraylib.a -o cpyquaticus.so


// Compile for internal renderer
// Clang cpyquaticus.c  -L lib/ -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL lib/libraylib.a -o cpyquaticus


int main(int argc, char* argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <field_width> <field_height> <num_agents> <num_steps> <render>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    // Main Method Arguments:
    // Field width (must bve divisible by 2 to ensure even field sizes)
    // Field height (must be divisible by 2 to ensure even field sizes)
    // Num Agents Per Team ()
    // Num Steps Per Episode:
    int field_width = atoi(argv[1]);
    int field_height = atoi(argv[2]);
    int num_agents = atoi(argv[3]);
    int num_steps = atoi(argv[4]);
    int render = atoi(argv[5]);
    time_t start, end;
    start = time(NULL);
    // Settings * create_game(int field_width, int field_height, int num_agents, int num_steps){
    // Force 2 team constraint for now
    settings * game = create_game(field_width, field_height, num_agents, 2, num_steps,1);
    episode * e = create_episode(game);
    if (render){
	int x = 1;
        //InitWindow(1200, 600, "C Maritime Capture the Flag");
        //SetTargetFPS(60);
        //initialize_game_starts(game, e);
        //for (int i = 0; i < num_steps; i++){
          //  BeginDrawing();
           // ClearBackground(BLACK);
            //int actions[2] = {2,17};//get_random_value(0,17)};

            //int actions[2] = {4,4};
            //for (int x = 0; x < num_agents; x++){
             //   drawPlayer(e->agents[x], field_width, field_height, 1200, 600);
            //}
            //for (int x = 0; x < 2; x++){
             //   drawFlag(game->flags[x], field_width, field_height, 1200, 600);
            //}
            //drawFieldLines(field_width, field_height, 1200, 600);
            //EndDrawing();
            //sleep(0.10);
            //step(game, e, actions);
            // for(int ff = 0; ff < game->num_agents; ff++){
            //     printf("Agent ID: %d Tag Penalty: %d OOB: %d\n",ff, e->rewards[ff]->tag,e->rewards[ff]->oob);
            // }
            //Update all agents 
            //Check for game events
            //Save new observations
       // }
        //CloseWindow();
    }
    else{
        for(int x = 0; x < 10000; x++){
        initialize_game_starts(game, e);
        real_obs * max = create_limit_obs(field_width+50.0f, field_height+50.0f, 360.0f, 3.0f, 60.0f, 1);
        real_obs * min = create_limit_obs(-50.0f, -50.0f, 0.0f, 0.0f, 0.0f, 0);
        int val;
        for (int i = 0; i < num_steps; i++){
            scanf("%d",&val);
            int actions[2] = {val,get_random_value(0,17)};
            // sleep(0.10);
            step(game, e, actions);
            float ** list = get_observations(game, e, 1, max, min);
            free_obs_list(list, game->num_agents);
            // for(int ff = 0; ff < game->num_agents; ff++){
            //     printf("Agent ID: %d Tag Penalty: %d\n",ff, e->rewards[ff]->tag);
            // }
            //Update all agents 
            //Check for game events
            //Save new observations
        }
        free(max);
        free(min);
    }
    free_episode(e, num_agents);
    free_game(game);
}
    end = time(NULL);
    printf("Elapsed Time: %f\n", difftime(end,start));
    return 0;
}

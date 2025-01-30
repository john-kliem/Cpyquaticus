#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>



typedef struct {
    float dt;
    float kp;
    float ki;
    float kd;
    float integral_max;
    float prev_error;
    float integral;
} PID;
const float heron_thrust_mapping[2][7] = {{-100.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0}, {-2.0, 0.0, 1.0, 2.0, 3.0, 5.0, 5.0}};
typedef struct {
    int agent_id;
    float max_speed;
    float speed_factor;
    float thrust_map[2][7]; 
    float max_thrust;
    float max_rudder;
    float turn_loss;
    float turn_rate;
    float max_acc;
    float max_dec;
    float dt;
    float thrust;
    float prev_pos[2];
    float pos[2];
    float speed;
    float heading;
    int has_flag;
    int on_their_side;
    float tagging_cooldown;
    int is_tagged;
    PID* speed_controller;
    PID* heading_controller;
} USV;

void print_pid_state(PID* pid);
void print_heron_state(USV* heron);

USV* create_heron(int agent_id) {
    USV* heron = (USV*)malloc(sizeof(USV)); // Cast the return value of malloc
    if (heron == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1); 
    }
    heron->agent_id = agent_id;
    heron->max_speed = 3.5;
    heron->speed_factor = 20.0;
    // Initialize thrust_map with appropriate heron values
    memcpy(heron->thrust_map, heron_thrust_mapping, sizeof(heron_thrust_mapping)); 
    heron->max_thrust = 70.0;
    heron->max_rudder = 100.0;
    heron->turn_loss = 0.85;
    heron->turn_rate = 70.0;
    heron->max_acc = 1.0;
    heron->max_dec = 1.0;
    heron->thrust = 0.0;
    heron->prev_pos[0] = 0.0;
    heron->prev_pos[1] = 0.0;
    heron->pos[0] = 0.0;
    heron->pos[1] = 0.0;
    heron->speed = 0.0;
    heron->heading = 0.0;
    heron->dt = 0.1;
    heron->speed_controller = (PID*)malloc(sizeof(PID));
    heron->heading_controller = (PID*)malloc(sizeof(PID));
    //Assign Controller Speed Configs
    heron->speed_controller->dt = 0.1;
    heron->speed_controller->kp = 1.0;
    heron->speed_controller->ki = 0.0;
    heron->speed_controller->kd = 0.0;
    heron->speed_controller->integral_max = 0.07;
    heron->speed_controller->prev_error = 0.0;
    heron->speed_controller->integral = 0.0;
    //Assign Controller Heading Configs
    heron->heading_controller->dt = 0.1;
    heron->heading_controller->kp = 0.35;
    heron->heading_controller->ki = 0.0;
    heron->heading_controller->kd = 0.07;
    heron->heading_controller->integral_max = 0.07;
    heron->heading_controller->prev_error = 0.0;
    heron->heading_controller->integral = 0.0;
    heron->has_flag = 0;
    heron->on_their_side = 1;
    heron->tagging_cooldown = 0.0;
    heron->is_tagged = 0;
    return heron;
}



void print_heron_state(USV* heron){
    printf("-- Current Heron State --\n");
    printf("Speed: %f\n",heron->speed);
    printf("Heading: %f\n",heron->heading);
    printf("X: %f\n", heron->pos[0]);
    printf("Y: %f\n",heron->pos[1]);
    printf("Prev X: %f\n",heron->prev_pos[0]);
    printf("Prev Y: %f\n",heron->prev_pos[1]);
    printf("thrust: %f\n", heron->thrust);
    printf("max_dec: %f\n", heron->max_dec);
    printf("max_acc: %f\n", heron->max_acc);
    printf("turn_rate: %f\n", heron->turn_rate);
    printf("turn_loss: %f\n", heron->turn_loss);
    printf("max_rudder: %f\n",heron->max_rudder);
    printf("max_thrust: %f\n",heron->max_thrust);
    printf("speed_factor: %f\n", heron->speed_factor);
    printf("max_speed: %f\n", heron->max_speed);
    printf("Has Flag: %d\n", heron->has_flag);
    printf("On Their Side: %d\n", heron->on_their_side);
    printf("Tagging Cooldown: %f\n", heron->tagging_cooldown);
    printf("Is Tagged: %d\n", heron->is_tagged);
    printf("Speed PID Controller\n");
    print_pid_state(heron->speed_controller);
    printf("Heading PID Controller\n");
    print_pid_state(heron->heading_controller);
}
void print_pid_state(PID* pid)
{
    float dt;
    float kp;
    float ki;
    float kd;
    float integral_max;
    float prev_error;
    float integral;
    printf("-- PID State --\n");
    printf("DT: %f\n", pid->dt);
    printf("kp: %f\n", pid->kp);
    printf("ki: %f\n",pid->ki);
    printf("kd: %f\n", pid->kd);
    printf("integral_max: %f\n",pid->integral_max);
    printf("prev_error: %f\n", pid->prev_error);
    printf("Integral: %f\n", pid->integral);
}

float min(float n1, float n2){
    if (n1 < n2){
        return n1;
    }
    return n2;
}

float controller_update(PID * pid, float error){
    pid->integral = min((pid->integral + error * pid->dt), pid->integral_max);
    float deriv = (error - pid->prev_error) / pid->dt;
    pid->prev_error = error;
    float p = pid->kp * error;
    float i = pid->ki * pid->integral;
    float d = pid->kd * deriv;
    return p + i + d;
}
float clip(float value, float lower_bound, float upper_bound){
    if (value < lower_bound){
        return lower_bound;
    }
    if (value > upper_bound){
        return upper_bound;
    }
    return value;
}
float interp(float x, const float xp[], const float fp[], int n){
    for (int i = 0; i < n-1; i++){
        if (x >= xp[i] && x < xp[i+1]){
            float t = (x - xp[i]) / (xp[i+1] - xp[i]);
            return fp[i] + t * (fp[i+1]-fp[i]);
        }
    }
    return 0.0;
}
float angle180(float deg){
    while (deg > 180){
        deg -= 360;
    }
    while (deg < -180){
        deg += 360;
    }
    return deg;
}
void move_heron(USV* heron, float desired_speed, float heading_error){
    float speed_error = desired_speed - heron->speed;
    desired_speed = controller_update(heron->speed_controller, speed_error);
    float desired_rudder = controller_update(heron->heading_controller, heading_error);
    float desired_thrust = heron->thrust + heron->speed_factor * desired_speed;
    desired_thrust = clip(desired_thrust, -1*heron->max_thrust, heron->max_thrust);
    desired_rudder = clip(desired_rudder, -1*heron->max_rudder, heron->max_rudder);

    float raw_speed = interp(desired_thrust, heron->thrust_map[0], heron->thrust_map[1], 7);
    float new_speed = min(raw_speed * 1 - ((fabs(desired_rudder)/100) * heron->turn_loss), heron->max_speed);

    if ((new_speed - heron->speed)/heron->dt > heron->max_acc){
        new_speed = heron->speed + heron->max_acc * heron->dt;
    }
    else if ((heron->speed - new_speed) / heron->dt > heron->max_dec){
        new_speed = heron->speed - heron->max_dec   * heron->dt;
    }
    float raw_d_hdg = desired_rudder * (heron->turn_rate / 100.0) * heron->dt;
    float thrust_d_hdg = raw_d_hdg * (1 + (fabs(desired_thrust)-50) / 50);

    heron->thrust = desired_thrust;

    //If not moving then shouldn't be able to turn
    if ((new_speed + heron->speed)/2.0 < 0.5){
        thrust_d_hdg = 0.0;
    }
    float new_heading = angle180(heron->heading + thrust_d_hdg);
    float hdg_rad = heron->heading * (M_PI/180.0);
    float new_hdg_rad = new_heading * (M_PI/180.0);
    float avg_speed = (new_speed + heron->speed) / 2.0;

    float s = sin(new_hdg_rad) + sin(hdg_rad);
    float c = cos(new_hdg_rad) + cos(hdg_rad);
    float avg_hdg = atan2(s,c);

    heron->prev_pos[0] = heron->pos[0];
    heron->prev_pos[1] = heron->pos[1];

    heron->pos[0] = heron->pos[0] + sin(avg_hdg) * avg_speed * heron->dt;
    heron->pos[1] = heron->pos[1] + cos(avg_hdg) * avg_speed * heron->dt;
    heron->speed = clip(new_speed, 0.0, heron->max_speed);
    heron->heading = angle180(new_heading);
    return;
}

void free_heron(USV* heron) {
    free(heron->speed_controller);
    free(heron->heading_controller); 
    free(heron);
}

PID* deep_copy_pid(PID* original) {
    if (original == NULL) {
        return NULL;
    }

    PID* copy = (PID*)malloc(sizeof(PID));
    if (copy == NULL) {
        fprintf(stderr, "Memory allocation failed for PID copy!\n");
        exit(1);
    }
    *copy = *original;
    return copy;
}



USV* deep_copy_heron(USV* original) {
    if (original == NULL) {
        return NULL;
    }

    USV* copy = (USV*)malloc(sizeof(USV));
    if (copy == NULL) {
        fprintf(stderr, "Memory allocation failed for USV copy!\n");
        exit(1);
    }

    // Copy primitive values
    *copy = *original;

    // Deep copy the thrust_map
    memcpy(copy->thrust_map, original->thrust_map, sizeof(original->thrust_map));

    // Deep copy speed_controller
    copy->speed_controller = (PID*)malloc(sizeof(PID));
    if (copy->speed_controller == NULL) {
        fprintf(stderr, "Memory allocation failed for speed controller!\n");
        free(copy);
        exit(1);
    }
    *copy->speed_controller = *original->speed_controller;

    // Deep copy heading_controller using helper function
    copy->heading_controller = deep_copy_pid(original->heading_controller);
    if (copy->heading_controller == NULL) {
        free(copy->speed_controller);
        free(copy);
        exit(1);
    }

    return copy;
}


int main() {
    USV* h = create_heron(0);
    print_heron_state(h);
    USV* copy = deep_copy_heron(h);
    printf("Copied Heron");
    move_heron(h, 2.5, 0.0);
    move_heron(h, 2.5, 0.0);
    move_heron(h, 2.5, 0.0);
    move_heron(h, 2.5, 0.0);
    move_heron(h, 2.5, 0.0);

    printf("\n\n\n");
    print_heron_state(h);
    printf("Created Heron\n");
    printf("Copied Heron State\n");
    print_heron_state(copy);
    free_heron(copy);
    free_heron(h);
    printf("Freed Herons\n");
    return 0; // Add return 0 to indicate successful program execution
}



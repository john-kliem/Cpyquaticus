#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct{
    int team;
    float pos[2];
    int grabbed;
    int grabbed_by;
} flag;

//Assumes by default the flag can't be grabbed on the spawn state
flag * create_flag(int team, int x, int y){
    flag* f = (flag*)malloc(sizeof(flag));
    f->team = team;
    f->pos[0] = x;
    f->pos[1] = y;
    f->grabbed = 0;
    f->grabbed_by = -1;
    return f;
}
flag * create_flag_full(int team, int x, int y, int grabbed, int grabbed_by){
    flag * f = create_flag(team, x, y);
    f->grabbed = grabbed;
    f->grabbed_by = grabbed_by;
    return f;
}

void free_flag(flag* f){
    // free(f->pos);
    free(f);
}

// int main(){
//     flag * f = create_flag(0, 0.0, 0.0);
//     free_flag(f);
// }
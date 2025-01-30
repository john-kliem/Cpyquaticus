#include "include/raylib.h"
#include "include/raymath.h"
#include <math.h>

#define WINDOW_WIDTH  800
#define WINDOW_HEIGHT 600
#define FIELD_WIDTH   40.0f  // Logical field width
#define FIELD_HEIGHT  30.0f  // Logical field height
#define FIELD_PADDING 50.0f
typedef struct {
    float x, y;    // Position coordinates (floating-point)
    float heading; // Heading in degrees
} Player;

typedef struct {
    float x, y; // Flag position (floating-point)
} Flag;

// Convert logical coordinates to screen coordinates
// Vector2 worldToScreen(float x, float y, float fw, float fh, float ww, float wh) {
//     Vector2 screenPos;
//     screenPos.x = ((x+FIELD_PADDING) / (fw+2*FIELD_PADDING)) * ww;
//     screenPos.y = wh - ((y+FIELD_PADDING / (fh+2*FIELD_PADDING)) * wh);
//     return screenPos;
// }
Vector2 worldToScreen(float wx, float wy, float field_width, float field_height, float ww, float wh) {
    float scaleX = ww / (field_width + 2 * FIELD_PADDING);
    float scaleY = wh / (field_height + 2 * FIELD_PADDING);
    Vector2 screenPos;
    screenPos.x = (wx + FIELD_PADDING) * scaleX;
    screenPos.y = wh - (wy + FIELD_PADDING) * scaleY;
    return screenPos;
}

// Draw a player with a directional arrow
void drawPlayer(USV * h, float fw, float fh, float ww, float wh) {
    Vector2 position = worldToScreen(h->pos[0], h->pos[1], fw, fh, ww, wh);
    DrawCircleV(position, 10, GREEN);

    // Draw directional arrow
    float rad = (fmodf(h->heading+360.0f, 360.0f)-90.0f) * (PI / 180.0f);
    Vector2 arrowEnd = {
        position.x + cosf(rad) * 20,
        position.y + sinf(rad) * 20
    };

    DrawLineV(position, arrowEnd, WHITE);
}

void drawFieldLines(float field_width, float field_height, float ww, float wh) {
    
    Vector2 start = worldToScreen(0, 0, field_width,field_height, ww, wh);
    Vector2 end = worldToScreen(0, field_height, field_width,field_height, ww, wh);
    DrawLineV(start, end, DARKGRAY);
    start = worldToScreen(field_width, 0, field_width,field_height, ww, wh);
    end = worldToScreen(field_width, field_height, field_width,field_height, ww, wh);
    DrawLineV(start, end, DARKGRAY);

    start = worldToScreen(0, 0, field_width,field_height, ww, wh);
    end = worldToScreen(field_width, 0, field_width,field_height, ww, wh);
    DrawLineV(start, end, DARKGRAY);
    start = worldToScreen(0, field_height, field_width,field_height, ww, wh);
    end = worldToScreen(field_width, field_height, field_width,field_height, ww, wh);
    DrawLineV(start, end, DARKGRAY);

    start = worldToScreen(field_width/2, 0, field_width,field_height, ww, wh);
    end = worldToScreen(field_width/2, field_height, field_width,field_height, ww, wh);
    DrawLineV(start, end, DARKGRAY);
   
}

// Draw a flag as a red square
void drawFlag(flag* f, float fw, float fh, float ww, float wh) {
    Vector2 position = worldToScreen(f->pos[0], f->pos[1],fw,fh,ww,wh);
    if(f->team){
        DrawRectangleV((Vector2){position.x - 5, position.y - 5}, (Vector2){10, 10}, BLUE);
    }
    else{
        DrawRectangleV((Vector2){position.x - 5, position.y - 5}, (Vector2){10, 10}, RED);
    }
}

// int main() {
//     InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "2D Float Coordinate Renderer");
//     SetTargetFPS(60);

//     // Define players and flags using float coordinates
//     Player players[] = {
//         {5.5f,  3.2f,  45},  // Player at (5.5,3.2), heading NE
//         {20.0f, 15.7f, 90},  // Player at (20.0,15.7), heading E
//         {35.3f, 10.8f, 270}, // Player at (35.3,10.8), heading W
//         {10.1f, 28.4f, 180}  // Player at (10.1,28.4), heading S
//     };

//     Flag flags[] = {
//         {15.5f, 5.5f},  // Flag at (15.5, 5.5)
//         {30.2f, 20.3f}  // Flag at (30.2, 20.3)
//     };

//     int playerCount = sizeof(players) / sizeof(players[0]);
//     int flagCount = sizeof(flags) / sizeof(flags[0]);

//     while (!WindowShouldClose()) {
//         BeginDrawing();
//         ClearBackground(BLACK);

//         // Draw flags
//         for (int i = 0; i < flagCount; i++) {
//             drawFlag(flags[i]);
//         }

//         // Draw players
//         for (int i = 0; i < playerCount; i++) {
//             drawPlayer(players[i]);
//         }

//         DrawText("Press ESC to exit", 10, 10, 20, WHITE);
//         EndDrawing();
//     }

//     CloseWindow();
//     return 0;
// }
// run: Clang render_test.c  -L lib/ -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL lib/libraylib.a -o rt

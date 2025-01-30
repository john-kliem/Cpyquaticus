#include "include/raylib.h"
#include "include/raymath.h"
#include <math.h>

#define WINDOW_WIDTH  800
#define WINDOW_HEIGHT 600
#define FIELD_WIDTH   40.0f  // Logical field width
#define FIELD_HEIGHT  30.0f  // Logical field height

typedef struct {
    float x, y;    // Position coordinates (floating-point)
    float heading; // Heading in degrees
} Player;

typedef struct {
    float x, y; // Flag position (floating-point)
} Flag;

// Convert logical coordinates to screen coordinates
Vector2 worldToScreen(float x, float y, float fw, float fh, float ww, float wh) {
    Vector2 screenPos;
    screenPos.x = (x / fw) * ww;
    screenPos.y = (y / fh) * wh;
    return screenPos;
}

// Draw a player with a directional arrow
void drawPlayer(USV * h, float fw, float fh, float ww, float wh) {
    Vector2 position = worldToScreen(p.x, p.y, fw, fh, ww, wh);
    DrawCircleV(position, 10, GREEN);

    // Draw directional arrow
    float rad = p.heading * (PI / 180.0f);
    Vector2 arrowEnd = {
        position.x + cosf(rad) * 20,
        position.y + sinf(rad) * 20
    };

    DrawLineV(position, arrowEnd, WHITE);
}

// Draw a flag as a red square
void drawFlag(flag* f, float fw, float fh, float ww, float wh) {
    Vector2 position = worldToScreen(f.x, f.y,fw,fh,ww,wh);
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

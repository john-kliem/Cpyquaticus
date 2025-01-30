#include <stdio.h>
#include <stdlib.h>

// Define a nested struct
typedef struct {
    int x, y; // Coordinates
} Point;

// Define the main struct
typedef struct {
    int id;
    Point *location; // Pointer to a dynamically allocated Point
} Shape;

int main() {
    int num_shapes;

    // Ask user for the number of shapes
    printf("Enter the number of shapes: ");
    scanf("%d", &num_shapes);

    // Dynamically allocate memory for the list of shapes
    Shape *shapes = (Shape *)malloc(num_shapes * sizeof(Shape));

    // Initialize and populate the list
    for (int i = 0; i < num_shapes; i++) {
        shapes[i].id = i + 1; // Assign ID

        // Allocate memory for each shape's location
        shapes[i].location = (Point *)malloc(sizeof(Point));

        // Assign random or user-defined values to the Point
        shapes[i].location->x = i * 10; // Example: x-coordinate
        shapes[i].location->y = i * 20; // Example: y-coordinate

        // Print the shape data
        printf("Shape %d: Location (%d, %d)\n", shapes[i].id, shapes[i].location->x, shapes[i].location->y);
    }

    // Free allocated memory
    for (int i = 0; i < num_shapes; i++) {
        free(shapes[i].location); // Free each shape's location
    }
    free(shapes); // Free the shape list

    return 0;
}

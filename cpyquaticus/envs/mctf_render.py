import pygame
import math

FIELD_PADDING = 50  # Adjust as needed
DARK_GRAY = (80, 80, 80)

class MCTFRender:
    def __init__(self, field_width=160, field_height=80, field_padding=50, screen_width=800, screen_height=600):
        pygame.init()
        self.field_width = field_width
        self.field_height = field_height
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("MCTF Game")
        self.clock = pygame.time.Clock()
        self.running = True
        self.objects = []  # List of objects to render

    def world_to_screen(self, wx, wy):
        """Convert world coordinates to screen coordinates."""
        scale_x = self.screen_width / (self.field_width + 2 * FIELD_PADDING)
        scale_y = self.screen_height / (self.field_height + 2 * FIELD_PADDING)
        
        screen_x = (wx + FIELD_PADDING) * scale_x
        screen_y = self.screen_height - (wy + FIELD_PADDING) * scale_y  # Invert Y for screen

        return pygame.math.Vector2(screen_x, screen_y)

    def add_object(self, x, y, heading, color=(255, 255, 255), size=10):
        """Add an object to be rendered at (x, y) with a given heading, color, and size."""
        self.objects.append({'pos': (x, y), 'heading': heading, 'color': color, 'size': size})

    def draw_objects(self):
        """Draw all added objects."""
        for obj in self.objects:
            position = self.world_to_screen(obj['pos'][0], obj['pos'][1])
            pygame.draw.circle(self.screen, obj['color'], (int(position.x), int(position.y)), obj['size']//2)
    def draw_field_lines(self):
        """Draw field boundary and center lines."""
        lines = [
            ((0, 0), (0, self.field_height)),  # Left vertical
            ((self.field_width, 0), (self.field_width, self.field_height)),  # Right vertical
            ((0, 0), (self.field_width, 0)),  # Top horizontal
            ((0, self.field_height), (self.field_width, self.field_height)),  # Bottom horizontal
            ((self.field_width / 2, 0), (self.field_width / 2, self.field_height)),  # Center vertical
        ]

        for start, end in lines:
            start_pos = self.world_to_screen(start[0], start[1])
            end_pos = self.world_to_screen(end[0], end[1])
            pygame.draw.line(self.screen, DARK_GRAY, start_pos, end_pos, 2)
    def run(self):
        """Main loop to run the renderer."""
        self.screen.fill((0, 0, 0))  # Clear screen with black
        self.draw_field_lines()
        self.draw_objects()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

# Example usage
if __name__ == "__main__":
    renderer = MCTFRender()
    for i in range(10000):
        renderer.run()
    renderer.close()

    

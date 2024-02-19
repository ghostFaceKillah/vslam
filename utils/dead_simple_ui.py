import attr
import numpy as np
import pygame

@attr.define
class DeadSimpleUiState:
    keys_pressed: list[int] = attr.Factory(list)
    mouse_clicks: list[tuple[int, int]] = attr.Factory(list)

@attr.define
class DeadSimpleUI:
    """ Thanks for the name, Peter. """
    screen: pygame.Surface
    clock: pygame.time.Clock
    fps: int = 30
    screen_width: int = 800
    screen_height: int = 600
    keys_pressed: set[int] = attr.Factory(set)

    @classmethod
    def make(
        cls,
        screen_width: int = 800,
        screen_height: int = 600,
        title: str = "Simple UI",
     ):
        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(title)
        clock = pygame.time.Clock()
        return cls( screen, clock , screen_width, screen_height)

    def update_image_and_get_state(self, numpy_img: np.ndarray) -> DeadSimpleUiState:
        # Convert numpy image to Pygame Surface and display it
        image_surface = pygame.surfarray.make_surface(numpy_img.transpose((1, 0, 2)))
        self.screen.fill((255, 255, 255))  # Fill the screen with white
        self.screen.blit(image_surface, (0, 0))  # Draw the image
        pygame.display.flip()

        mouse_clicks = []

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()  # Exit the program
            elif event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)
            elif event.type == pygame.KEYUP:
                self.keys_pressed.remove(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_clicks.append(event.pos)

        # Limit the frame rate to 30 FPS
        state = DeadSimpleUiState(
            keys_pressed=list(self.keys_pressed),
            mouse_clicks=mouse_clicks
        )

        self.clock.tick(30)

        return state



if __name__ == '__main__':
    # Example usage
    ui = DeadSimpleUI.make()
    while True:
        state = ui.update_image_and_get_state(np.zeros((600, 800, 3), dtype=np.uint8))
        print(f"Keys pressed: {state.keys_pressed}, Mouse clicks: {state.mouse_clicks}")
        # Optionally, check for a condition to break the loop



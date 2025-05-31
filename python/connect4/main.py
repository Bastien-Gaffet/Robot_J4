from core import init_game
from camera.camera import camera_loop

if __name__ == "__main__":
    game_state = init_game()
    camera_loop(game_state)
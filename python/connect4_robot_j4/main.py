from core import init_game
from game_loop import run_game_loop

if __name__ == "__main__":
    game_state = init_game()
    run_game_loop(game_state)
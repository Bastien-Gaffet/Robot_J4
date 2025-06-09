def main():
    from connect4_robot_j4.core import init_game
    from connect4_robot_j4 import run_game_loop

    game_state, game_data = init_game()
    run_game_loop(game_state, game_data)

if __name__ == "__main__":
    main()
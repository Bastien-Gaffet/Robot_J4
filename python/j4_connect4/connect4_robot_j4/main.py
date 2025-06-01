def main():
    from connect4_robot_j4.core import init_game
    from connect4_robot_j4.game_loop import run_game_loop

    game_state = init_game()
    run_game_loop(game_state)

if __name__ == "__main__":
    main()
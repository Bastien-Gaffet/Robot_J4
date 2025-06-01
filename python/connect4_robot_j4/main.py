def main():
    from core import init_game
    from game_loop import run_game_loop

    game_state = init_game()
    run_game_loop(game_state)

if __name__ == "__main__":
    main()
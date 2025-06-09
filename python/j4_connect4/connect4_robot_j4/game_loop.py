import cv2
import time
import pygame
import datetime
import connect4_robot_j4.constants as cs
from connect4_robot_j4.core import init_game
from connect4_robot_j4.camera import (
    detect_tokens,
    overlay_on_camera,
    stabilize_grid,
    grid_to_matrix,
    is_valid_move,
    matrices_are_different,
    update_player_matrices,
    mouse_callback,
    is_valid_game_move,
    is_valid_new_move,
    add_column_to_database,
)
from connect4_robot_j4.minimax import (
    afficher_message,
    confirmer_coup_ia,
    placer_jeton,
    plateau_plein,
    time_to_play,
    verifier_victoire,
)
from connect4_robot_j4.arduino_serial import send_to_arduino, serial_obj
from connect4_robot_j4.camera import initialize_camera
from connect4_robot_j4.firebase_db import initialize_firebase, send_game_data

def detect_game_start(current_matrix, game_state):
    # Initialization phase
    if game_state.initialization_phase:
        if all(all(cell == 0 for cell in row) for row in current_matrix):
            print("Initialization successful - empty grid confirmed")
            print("Initial empty matrix:")
            for row in current_matrix:
                print(row)
            print("---------------------")
            game_state.last_stable_matrix = [row[:] for row in current_matrix]
            game_state.last_change_time = time.time()

            if serial_obj is not None:
                send_to_arduino(serial_obj, game_state.joueur_courant + 7)
            else:
                print("Arduino connection error - The game will continue without serial.")

            print("The first player is", game_state.joueur_courant)
            # The token pickup is initialized
            send_to_arduino(serial_obj, 13)

            # If the AI starts, make its first move
            if game_state.joueur_courant == 1:
                send_to_arduino(serial_obj, 8)
                send_to_arduino(serial_obj, 12)
                time_to_play(game_state)
            else:
                send_to_arduino(serial_obj, 9)

            game_state.initialization_phase = False
            return True
        else:
            print("Waiting for empty grid to start game...")
    return False

def check_victory(player, game_state, game_data):
    # Check if there is a victory
    if verifier_victoire(player):
        game_state.game_over = True
        message = "Congratulations! You won!" if player == 2 else "The computer won!"
        afficher_message(message)
        send_to_arduino(serial_obj, 22 if player == 2 else 21)
        game_data.game_end_time = datetime.datetime.now()
        game_data.winner = who_wins(player)  # à adapter à ton code
        db = initialize_firebase()
        send_game_data(game_data, db)
        pygame.time.delay(3000)
        return
    elif plateau_plein():
        game_state.game_over = True
        afficher_message("Draw!")
        send_to_arduino(serial_obj, 20)
        game_data.game_end_time = datetime.datetime.now()
        game_data.winner = "Draw"
        db = initialize_firebase()
        send_game_data(game_data, db)
        pygame.time.delay(3000)
        return
    # Switch between players (1 → 2, 2 → 1)
    game_state.joueur_courant = 3 - player
    print(f"Player's turn {game_state.joueur_courant}")

    send_to_arduino(serial_obj, game_state.joueur_courant + 7)
    if game_state.joueur_courant == 1:
        send_to_arduino(serial_obj, 12)

def who_wins(player):
    if player == 1:
        return "AI (Red)"
    elif player == 2:
        return "Player (Yellow)"
    return None

def update_from_camera(current_matrix, previous_matrix, game_state, game_data):
    #Updates the board with camera data and handles the game logic.  
    # Check if the move is valid according to the game rules
    if game_state.game_over:
        return False

    is_valid, player, column = is_valid_game_move(current_matrix, previous_matrix, game_state)
    if not is_valid:
        return False
    if not is_valid_new_move(previous_matrix, current_matrix):
        return False
    
    # If waiting for AI move detection and the AI has indeed played
    if game_state.en_attente_detection and player == 1:
        confirmer_coup_ia(game_state)
        print(f"AI move detected in column {column + 1}")
   
    # Update the Pygame display
    placer_jeton(column, player)
    pygame.display.update()

    add_column_to_database(current_matrix, previous_matrix, game_data)
    check_victory(player, game_state, game_data)

    pygame.display.update()

    # If it's the AI's turn, make the AI play
    if not game_state.game_over and game_state.joueur_courant == 1:
        time_to_play(game_state)

    return True

def run_game_loop(game_state, game_data):
    # Try to initialize the camera
    camera = initialize_camera()

    # Read a frame from the camera or create a fallback image
    ret, frame = camera.get_frame()
    print("Camera initialized!")

    # Display the result
    cv2.imshow("Camera Preview", frame)
    cv2.destroyAllWindows()

    # Main loop
    while True:
        # Handle Pygame events to prevent the interface from appearing frozen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                camera.release()
                cv2.destroyAllWindows()
                pygame.quit()
                return

        ret, frame = camera.get_frame()
        if not ret:
            print("Error: Unable to read a frame from the camera.")
            break

        # Flip the image horizontally
        frame = cv2.flip(frame, 1)

        # Token detection every frame
        current_grid = detect_tokens(frame)

        # Buffer update every frame
        game_state.grid_buffer.append(current_grid)
        if len(game_state.grid_buffer) > cs.BUFFER_SIZE:
            game_state.grid_buffer.pop(0)

        current_time = time.time()
        game_state.grid_changed = False

        # Stabilization and conversion to matrix only once per interval
        if current_time - game_state.last_grid_update_time >= cs.GRID_UPDATE_INTERVAL:
            if len(game_state.grid_buffer) >= cs.BUFFER_SIZE // 2:
                stable_grid = stabilize_grid(current_grid, game_state)
                current_matrix = grid_to_matrix(stable_grid)

                # Check that the physical structure of the grid is valid
                structure_valid = game_state.last_stable_matrix is None or is_valid_move(game_state.last_stable_matrix, current_matrix)

                if structure_valid:
                    # If a change is detected and the stabilization time has passed
                    if (game_state.last_stable_matrix is None or
                        matrices_are_different(current_matrix, game_state.last_stable_matrix)) and \
                    (current_time - game_state.last_change_time >= cs.SETTLING_TIME):

                        # Attempt to update the game with this new grid
                        detect_game_start(current_matrix, game_state)
                        game_updated = update_from_camera(current_matrix, game_state.last_stable_matrix, game_state, game_data)

                        # Only if the game was successfully updated, update the stable matrix
                        if game_updated:
                            game_state.grid_changed = True
                            game_state.current_matrix = current_matrix

                            # Update the player-specific matrices
                            update_player_matrices(current_matrix, game_state.last_stable_matrix)

                            # Deep copy to avoid shared references
                            game_state.last_stable_matrix = [row[:] for row in current_matrix]
                            game_state.last_change_time = current_time

                            # Display of the newly validated grid
                            print("New grid detected and validated:")
                            for row in current_matrix:
                                print(row)
                            print("---------------------")

                game_state.last_grid_update_time = current_time

        # Use the latest stable grid for display
        display_grid = game_state.last_stable_grid if game_state.last_stable_grid is not None else current_grid
        camera_overlay = overlay_on_camera(frame, display_grid)

        # Display the current status
        status_text = "Grid modified!" if game_state.grid_changed else "Stable grid"
        status_color = (0, 0, 255) if game_state.grid_changed else (0, 255, 0)
        cv2.putText(camera_overlay, status_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Display the remaining time until the next update
        time_to_next = max(0, cs.GRID_UPDATE_INTERVAL - (current_time - game_state.last_grid_update_time))
        cv2.putText(camera_overlay, f"Next update : {time_to_next:.1f}s",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the current player
        player_text = f"Turn: {'AI (Red)' if game_state.joueur_courant == 1 else 'Player (Yellow)'}"
        cv2.putText(camera_overlay, player_text,
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if game_state.joueur_courant == 1 else (0, 255, 255), 2)

        cv2.imshow("Camera Feed", camera_overlay)
        cv2.setMouseCallback("Camera Feed", lambda event, x, y, flags, param: mouse_callback(event, x, y, flags, param, frame))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Reset the game
            print("Resetting game...")
            game_state, game_data = init_game()

    # Release the resources
    camera.release()
    cv2.destroyAllWindows()
    pygame.quit()

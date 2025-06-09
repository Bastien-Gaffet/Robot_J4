import cv2
import math
import time
import connect4_robot_j4.constants as cs
from collections import Counter
from connect4_robot_j4.minimax import verifier_coup_ia
import numpy as np

def detect_circles(frame, lower, upper):
    # Detects circles of a specific color in the image
    roi = frame[cs.ROI_Y:cs.ROI_Y + cs.ROI_H, cs.ROI_X:cs.ROI_X + cs.ROI_W]
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cs.KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cs.KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if cs.MIN_AREA <= area <= cs.MAX_AREA:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circularity = area / (math.pi * (radius ** 2))
            if circularity >= cs.MIN_CIRCULARITY:
                centers.append((int(cx), int(cy)))

    return centers, mask

def detect_tokens(frame):
    # Detect all the red and yellow tokens in the image
    red_centers1, _ = detect_circles(frame, cs.LOWER_RED1, cs.UPPER_RED1)
    red_centers2, _ = detect_circles(frame, cs.LOWER_RED2, cs.UPPER_RED2)
    red_centers = red_centers1 + red_centers2

    yellow_centers1, _ = detect_circles(frame, cs.LOWER_YELLOW1, cs.UPPER_YELLOW1)
    yellow_centers2, _ = detect_circles(frame, cs.LOWER_YELLOW2, cs.UPPER_YELLOW2)
    yellow_centers3, _ = detect_circles(frame, cs.LOWER_YELLOW3, cs.UPPER_YELLOW3)
    yellow_centers4, _ = detect_circles(frame, cs.LOWER_YELLOW4, cs.UPPER_YELLOW4)
    yellow_centers = yellow_centers1 + yellow_centers2 + yellow_centers3 + yellow_centers4

    # Creating an empty grid
    grid = {}

    # Define the cell dimensions
    cell_width = cs.ROI_W / cs.COLS
    cell_height = cs.ROI_H / cs.ROWS

    # Process each red token
    for cx, cy in red_centers:
        # Convert the coordinates to grid indices
        col = int(cx / cell_width)
        row = int(cy / cell_height)

        # Ensure the indices are within bounds
        if 0 <= row < cs.ROWS and 0 <= col < cs.COLS:
            grid[(row, col)] = "red"

    # Process each yellow token
    for cx, cy in yellow_centers:
        col = int(cx / cell_width)
        row = int(cy / cell_height)
        if 0 <= row < cs.ROWS and 0 <= col < cs.COLS:
            grid[(row, col)] = "yellow"

    return grid

def overlay_on_camera(frame, grid):
    # Overlay the detected grid on the camera image
    overlay = frame.copy()

    # Draw the grid for visualization
    for row in range(cs.ROWS):
        for col in range(cs.COLS):
            # Calculate the center of each cell
            cx = cs.ROI_X + int((col + 0.5) * (cs.ROI_W / cs.COLS))
            cy = cs.ROI_Y + int((row + 0.5) * (cs.ROI_H / cs.ROWS))

            # Draw the cell frame
            cell_w = int(cs.ROI_W / cs.COLS)
            cell_h = int(cs.ROI_H / cs.ROWS)
            cv2.rectangle(overlay,
                         (cs.ROI_X + col * cell_w, cs.ROI_Y + row * cell_h),
                         (cs.ROI_X + (col + 1) * cell_w, cs.ROI_Y + (row + 1) * cell_h),
                         (100, 100, 100), 1)

            # If a token is present in this cell, draw it
            if (row, col) in grid:
                color = grid[(row, col)]
                color_bgr = (0, 0, 255) if color == "red" else (0, 255, 255) if color == "yellow" else (255, 255, 255)
                cv2.circle(overlay, (cx, cy), int(min(cell_w, cell_h) * 0.4), color_bgr, -1)

    # Draw the frame ROI
    cv2.rectangle(overlay, (cs.ROI_X, cs.ROI_Y), (cs.ROI_X + cs.ROI_W, cs.ROI_Y + cs.ROI_H), (0, 255, 0), 2)

    return overlay

def stabilize_grid(current_grid, game_state):
    # Stabilizes the grid by analyzing the buffer of grids and applying a majority vote
    # If the buffer is not full, return the current grid
    if len(game_state.grid_buffer) < cs.BUFFER_SIZE:
        return current_grid

    # Create a stabilized grid
    stable_grid = {}
    all_positions = set()

    # Collect all positions from the grid buffer
    for grid in game_state.grid_buffer:
        all_positions.update(grid.keys())

    # Iterate through all positions in the buffer
    for pos in all_positions:
        # Collect the colors from all grids in the buffer for this position
        colors = [grid.get(pos, None) for grid in game_state.grid_buffer]
        colors = [c for c in colors if c is not None]  # Remove None values

        # If no color is detected for this position, skip it
        if not colors:
            continue

        # Count the occurrences of each color
        color_counts = Counter(colors)
        most_common_color, count = color_counts.most_common(1)[0]

        # If the most common color appears in at least DETECTION_THRESHOLD of the grids, keep it
        if count / len(game_state.grid_buffer) >= cs.DETECTION_THRESHOLD:  # Use the full buffer as the denominator
            stable_grid[pos] = most_common_color

    # Check that the grid is valid according to the rules
    if not is_valid_grid(stable_grid):
        print("Invalid grid, ignored")
        # Keep the previous grid if the new one is not valid
        if game_state.last_stable_grid is not None:
            return game_state.last_stable_grid

    # Update the stabilization timestamp
    game_state.last_stabilization_time = time.time()
    game_state.last_stable_grid = stable_grid

    return stable_grid

def grid_to_matrix(grid):
    # Convert the grid representation from a dictionary to a 2D matrix
    # Create an empty matrix filled with zeros
    matrix = [[0 for _ in range(cs.COLS)] for _ in range(cs.ROWS)]

    # Fill the matrix with the values from the grid dictionary
    for (row, col), color in grid.items():
        if 0 <= row < cs.ROWS and 0 <= col < cs.COLS:  # Check the boundaries
            if color == "red":
                matrix[row][col] = 1
            elif color == "yellow":
                matrix[row][col] = 2

    return matrix

def is_valid_grid(grid):
    # Check that the grid complies with the Connect Four gravity rules
    # Convert to a matrix to facilitate verification
    matrix = [[0 for _ in range(cs.COLS)] for _ in range(cs.ROWS)]

    for (row, col), color in grid.items():
        if 0 <= row < cs.ROWS and 0 <= col < cs.COLS:
            if color == "red":
                matrix[row][col] = 1
            elif color == "yellow":
                matrix[row][col] = 2

    # Verify the gravity rules
    for col in range(cs.COLS):
        # For each column, check from bottom to top
        found_empty = False
        for row in range(cs.ROWS-1, -1, -1):  # From bottom to top
            if matrix[row][col] == 0:  # Empty cell
                found_empty = True
            elif found_empty:  # If a token is found after an empty cell (gravity violation)
                return False

    return True

def is_valid_game_move(current_matrix, previous_matrix, game_state):
    # Check if the detected move is valid according to the game rules.
    # Determine the player and column of the last move
    last_player = get_last_player(current_matrix, previous_matrix)
    last_move = get_last_move_column(current_matrix, previous_matrix)

    # If no player is detected, no valid move
    if last_player is None or last_move == -1:
        return False, None, None

    # Convert last_player to an integer
    player_num = 1 if last_player == "red" else 2 if last_player == "yellow" else None

    # Check if it is indeed this player’s turn
    if player_num != game_state.joueur_courant:
        print(f"Detection ignored: it is the player's turn {game_state.joueur_courant}, but {player_num} has been detected")
        return False, None, None

    # Special check for the AI's move
    if game_state.en_attente_detection and player_num == 1:
        if not verifier_coup_ia(last_move, game_state):
            print(f"Move detected in column {last_move + 1} does not match the expected AI move")
            return False, None, None

    return True, player_num, last_move

def count_tokens(matrix):
    # Count the total number of tokens in the matrix
    count = 0
    for row in range(cs.ROWS):
        for col in range(cs.COLS):
            if matrix[row][col] > 0:  # A token is present (1 for red, 2 for yellow)
                count += 1
    return count

def is_valid_move(previous_matrix, current_matrix):
    if previous_matrix is None:
        # If it’s the first grid, it must be empty
        return all(all(cell == 0 for cell in row) for row in current_matrix)

    previous_count = count_tokens(previous_matrix)
    current_count = count_tokens(current_matrix)

    return current_count == previous_count + 1

def matrices_are_different(matrix1, matrix2):
    # Compare two matrices and return True if they are different
    if matrix1 is None or matrix2 is None:
        return True  # If either of the matrices is None, consider them different

    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            if matrix1[i][j] != matrix2[i][j]:
                return True  # Difference found

    return False  # No difference found

def get_last_move_column(current_matrix, previous_matrix):
    # Determine the column of the last move played by comparing two consecutive matrices
    # If either matrix is None, it is impossible to determine the last move
    if current_matrix is None or previous_matrix is None:
        return -1

    # Iterate through each column
    for col in range(cs.COLS):
        # Find the first differing token in this column (from bottom to top)
        for row in range(cs.ROWS-1, -1, -1):
            # If a token is found in the current matrix that wasn't there before
            if current_matrix[row][col] != 0 and previous_matrix[row][col] == 0:
                return col

    # No new token found
    return -1

def get_last_player(current_matrix, previous_matrix):
    # Determine which player made the last move
    # If either matrix is None, it is impossible to determine the last player
    if current_matrix is None or previous_matrix is None:
        return None

    # Iterate through each column
    for row in range(cs.ROWS):
        for col in range(cs.COLS):
            # If a token is found in the current matrix that wasn’t there before
            if previous_matrix[row][col] == 0 and current_matrix[row][col] != 0:
                # Identify the player based on the value
                if current_matrix[row][col] == 1:
                    return "red"
                elif current_matrix[row][col] == 2:
                    return "yellow"

    # No new token found
    return None

def is_valid_new_move(previous_matrix, current_matrix, empty_value=0):
    previous_matrix = np.array(previous_matrix)
    current_matrix = np.array(current_matrix)

    # Vérifie que les matrices ont la même forme
    if previous_matrix.shape != current_matrix.shape:
        print("Les matrices n'ont pas la même forme !")
        return False

    # Trouver les indices où les valeurs diffèrent
    differences = np.where(previous_matrix != current_matrix)

    # Vérifie que differences est bien un tuple de deux éléments non vides
    if len(differences) != 2 or len(differences[0]) != 1 or len(differences[1]) != 1:
        print("Invalid move detected.")
        return False

    y, x = differences[0][0], differences[1][0]
    
    # Vérifie que l'ancienne case était vide et la nouvelle occupée
    if previous_matrix[y, x] == empty_value and current_matrix[y, x] != empty_value:
        return True

    return False

def is_empty_matrix(matrix):
    # Check if the matrix is empty (no tokens placed)
    return all(all(cell == 0 for cell in row) for row in matrix)

def update_player_matrices(current_matrix, previous_matrix):
    # Update each player's matrices based on the detected change
    global last_red_move_matrix, last_yellow_move_matrix

    # If there is no previous matrix, it is impossible to determine the last player
    if previous_matrix is None:
        return

    # Find the newly added token
    for row in range(cs.ROWS):
        for col in range(cs.COLS):
            # If a token has been added
            if previous_matrix[row][col] == 0 and current_matrix[row][col] != 0:
                # Check the color of the added token
                if current_matrix[row][col] == 1:  # Red
                    # Copy the current matrix
                    last_red_move_matrix = [row[:] for row in current_matrix]
                elif current_matrix[row][col] == 2:  # Jaune
                    # copy the current matrix
                    last_yellow_move_matrix = [row[:] for row in current_matrix]
                return

def get_last_red_move_grid():
    # Return the grid as it was after the last move by the red player
    global last_red_move_matrix
    return last_red_move_matrix

def get_last_yellow_move_grid():
    # Return the grid as it was after the last move by the yellow player
    global last_yellow_move_matrix
    return last_yellow_move_matrix

def mouse_callback(event, x, y, flags, param, frame):
    # Callback for mouse clicks – useful for debugging colors
    if event == cv2.EVENT_LBUTTONDOWN:
        if cs.ROI_X <= x <= cs.ROI_X + cs.ROI_W and cs.ROI_Y <= y <= cs.ROI_Y + cs.ROI_H:
            local_x = x - cs.ROI_X
            local_y = y - cs.ROI_Y
            roi = frame[cs.ROI_Y:cs.ROI_Y + cs.ROI_H, cs.ROI_X:cs.ROI_X + cs.ROI_W]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[local_y, local_x]
            print(f"HSV at this point: H={h}, S={s}, V={v}")

def add_column_to_database(current_matrix, previous_matrix, game_data):
    """
    Adds a column move to the database for the current game state.
    """
    col = get_last_move_column(current_matrix, previous_matrix)
    if col != -1:
        game_data.moves.append(col + 1)  # de 0-6 vers 1-7

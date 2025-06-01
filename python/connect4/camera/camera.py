import cv2
import numpy as np
import math
import time
from collections import Counter
from minimax.minimax_functions import verifier_coup_ia

# Definition of HSV colors
LOWER_RED1 = np.array([0, 100, 80])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 100, 80])
UPPER_RED2 = np.array([180, 255, 255])

LOWER_YELLOW1 = np.array([15, 100, 100])
UPPER_YELLOW1 = np.array([30, 255, 255])
LOWER_YELLOW2 = np.array([30, 100, 100])
UPPER_YELLOW2 = np.array([45, 255, 255])

LOWER_YELLOW3 = np.array([29, 200, 130])
UPPER_YELLOW3 = np.array([36, 255, 200])

LOWER_YELLOW4 = np.array([25, 200, 110])
UPPER_YELLOW4 = np.array([33, 255, 200])

# Constants for image processing
KERNEL = np.ones((7, 7), np.uint8)

# Board settings
ROWS, COLS = 6, 7
ROI_X, ROI_Y, ROI_W, ROI_H = 50, 50, 500, 400
MIN_AREA = 300
MAX_AREA = 3000
MIN_CIRCULARITY = 0.6

# Stabilization settings
BUFFER_SIZE = 20
DETECTION_THRESHOLD = 0.6
SETTLING_TIME = 1.5  # Waiting time in seconds after a change
GRID_UPDATE_INTERVAL = 0.5  # Update interval in seconds

def detect_circles(frame, lower, upper):
    # Detects circles of a specific color in the image
    roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA <= area <= MAX_AREA:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circularity = area / (math.pi * (radius ** 2))
            if circularity >= MIN_CIRCULARITY:
                centers.append((int(cx), int(cy)))

    return centers, mask

def detect_tokens(frame):
    # Detect all the red and yellow tokens in the image
    red_centers1, _ = detect_circles(frame, LOWER_RED1, UPPER_RED1)
    red_centers2, _ = detect_circles(frame, LOWER_RED2, UPPER_RED2)
    red_centers = red_centers1 + red_centers2

    yellow_centers1, _ = detect_circles(frame, LOWER_YELLOW1, UPPER_YELLOW1)
    yellow_centers2, _ = detect_circles(frame, LOWER_YELLOW2, UPPER_YELLOW2)
    yellow_centers3, _ = detect_circles(frame, LOWER_YELLOW3, UPPER_YELLOW3)
    yellow_centers4, _ = detect_circles(frame, LOWER_YELLOW4, UPPER_YELLOW4)
    yellow_centers = yellow_centers1 + yellow_centers2 + yellow_centers3 + yellow_centers4

    # Creating an empty grid
    grid = {}

    # Define the cell dimensions
    cell_width = ROI_W / COLS
    cell_height = ROI_H / ROWS

    # Process each red token
    for cx, cy in red_centers:
        # Convert the coordinates to grid indices
        col = int(cx / cell_width)
        row = int(cy / cell_height)

        # Ensure the indices are within bounds
        if 0 <= row < ROWS and 0 <= col < COLS:
            grid[(row, col)] = "red"

    # Process each yellow token
    for cx, cy in yellow_centers:
        col = int(cx / cell_width)
        row = int(cy / cell_height)
        if 0 <= row < ROWS and 0 <= col < COLS:
            grid[(row, col)] = "yellow"

    return grid

def overlay_on_camera(frame, grid):
    # Overlay the detected grid on the camera image
    overlay = frame.copy()

    # Draw the grid for visualization
    for row in range(ROWS):
        for col in range(COLS):
            # Calculate the center of each cell
            cx = ROI_X + int((col + 0.5) * (ROI_W / COLS))
            cy = ROI_Y + int((row + 0.5) * (ROI_H / ROWS))

            # Draw the cell frame
            cell_w = int(ROI_W / COLS)
            cell_h = int(ROI_H / ROWS)
            cv2.rectangle(overlay,
                         (ROI_X + col * cell_w, ROI_Y + row * cell_h),
                         (ROI_X + (col + 1) * cell_w, ROI_Y + (row + 1) * cell_h),
                         (100, 100, 100), 1)

            # If a token is present in this cell, draw it
            if (row, col) in grid:
                color = grid[(row, col)]
                color_bgr = (0, 0, 255) if color == "red" else (0, 255, 255) if color == "yellow" else (255, 255, 255)
                cv2.circle(overlay, (cx, cy), int(min(cell_w, cell_h) * 0.4), color_bgr, -1)

    # Draw the frame ROI
    cv2.rectangle(overlay, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)

    return overlay

def stabilize_grid(current_grid, game_state):
    # Stabilizes the grid by analyzing the buffer of grids and applying a majority vote
    # If the buffer is not full, return the current grid
    if len(game_state.grid_buffer) < BUFFER_SIZE:
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
        if count / len(game_state.grid_buffer) >= DETECTION_THRESHOLD:  # Use the full buffer as the denominator
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
    matrix = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    # Fill the matrix with the values from the grid dictionary
    for (row, col), color in grid.items():
        if 0 <= row < ROWS and 0 <= col < COLS:  # Check the boundaries
            if color == "red":
                matrix[row][col] = 1
            elif color == "yellow":
                matrix[row][col] = 2

    return matrix

def is_valid_grid(grid):
    # Check that the grid complies with the Connect Four gravity rules
    # Convert to a matrix to facilitate verification
    matrix = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    for (row, col), color in grid.items():
        if 0 <= row < ROWS and 0 <= col < COLS:
            if color == "red":
                matrix[row][col] = 1
            elif color == "yellow":
                matrix[row][col] = 2

    # Verify the gravity rules
    for col in range(COLS):
        # For each column, check from bottom to top
        found_empty = False
        for row in range(ROWS-1, -1, -1):  # From bottom to top
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
    for row in range(ROWS):
        for col in range(COLS):
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
    for col in range(COLS):
        # Find the first differing token in this column (from bottom to top)
        for row in range(ROWS-1, -1, -1):
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
    for row in range(ROWS):
        for col in range(COLS):
            # If a token is found in the current matrix that wasn’t there before
            if previous_matrix[row][col] == 0 and current_matrix[row][col] != 0:
                # Identify the player based on the value
                if current_matrix[row][col] == 1:
                    return "red"
                elif current_matrix[row][col] == 2:
                    return "yellow"

    # No new token found
    return None

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
    for row in range(ROWS):
        for col in range(COLS):
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
        if ROI_X <= x <= ROI_X + ROI_W and ROI_Y <= y <= ROI_Y + ROI_H:
            roi_x, roi_y = x - ROI_X, y - ROI_Y
            roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[roi_y, roi_x]
            print(f"HSV at this point: H={h}, S={s}, V={v}")
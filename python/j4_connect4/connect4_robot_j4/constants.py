import numpy as np
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
import numpy as np
# Definition of HSV colors thresholds for color detection
LOWER_RED1 = np.array([0, 50, 50])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 50, 50])
UPPER_RED2 = np.array([180, 255, 255])


LOWER_YELLOW1 = np.array([15, 100, 100])
UPPER_YELLOW1 = np.array([30, 255, 255])
LOWER_YELLOW2 = np.array([30, 100, 100])
UPPER_YELLOW2 = np.array([45, 255, 255])
LOWER_YELLOW3 = np.array([26, 140, 200])
UPPER_YELLOW3 = np.array([32, 255, 255])
LOWER_YELLOW4 = np.array([25, 130, 200])
UPPER_YELLOW4 = np.array([33, 255, 255])

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

# Camera settings
MAX_INDEX = 3  # Maximum index for camera selection
IP_CAM_URL = "http://192.168.1.55:4747/video"
USE_IP_CAM = False  # Toggle this to False to disable IP cam check
PREFERRED_INDEX = None  # Default camera index, non used : None

#MiniMax settings
MINIMAX_DEPTH = 7  # Default depth for the Minimax algorithm
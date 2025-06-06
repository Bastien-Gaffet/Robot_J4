import cv2
import time
import numpy as np
import connect4_robot_j4.constants as cs
import socket
from urllib.parse import urlparse

class CameraHandler:
    def __init__(self, cap):
        self.cap = cap
        self.valid = cap is not None

    def get_frame(self):
        if self.cap is None:
            return False, create_fallback_frame()
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, create_fallback_frame()
        return True, frame

    def release(self):
        if self.cap:
            self.cap.release()

def is_ip_cam_available(ip, port, timeout=2):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False
    
def initialize_camera(use_ip_cam=cs.USE_IP_CAM, ip_cam_url=cs.IP_CAM_URL, preferred_index=None, max_index=cs.MAX_INDEX):
    #Tries to open IP camera first (if enabled), then local cameras by index.
    #Returns the first working VideoCapture object, or None if no camera is found.
    print("Initializing the camera...")
    if use_ip_cam:
        parsed_url = urlparse(ip_cam_url)
        ip = parsed_url.hostname
        port = parsed_url.port or 4747

        if is_ip_cam_available(ip, port):
            cap = cv2.VideoCapture(ip_cam_url)
            stabilize_camera(cap)
            if cap.isOpened():
                print(f"IP camera connected: {ip_cam_url}")
                return CameraHandler(cap)
            cap.release()
            print("Failed to connect to IP camera.")
        
        # If a preferred camera index is provided
    if preferred_index is not None:
        print(f"Trying preferred USB camera index: {preferred_index}")
        cap = cv2.VideoCapture(preferred_index)
        if cap.isOpened():
            print(f"Camera connected at preferred index {preferred_index}")
            return CameraHandler(cap)
        cap.release()
        print(f"Preferred camera index {preferred_index} not available.")

    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera detected at index {index}")
            return CameraHandler(cap)
        cap.release()
    print("No camera detected.")
    print("Error: Unable to open the camera. Please check the URL or the connection.")
    return CameraHandler(None)

def stabilize_camera(cap, timeout=10):
    print("Stabilizing camera...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        ret, frame = ret, frame = cap.read()
        if ret and frame is not None:
            print("First valid frame received.")
            return True
        time.sleep(0.3)  # Wait a bit before retrying to avoid flooding
    print("No valid frame received during stabilization.")
    return False

def create_fallback_frame(width=640, height=480, message="No camera detected"):
    #Creates a black image with a centered message to display when no camera is available.
    # Create a black image
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White in BGR
    thickness = 2
    
    # Get text size to center it
    (text_width, text_height), _ = cv2.getTextSize(message, font, font_scale, thickness)
    x = (width - text_width) // 2
    y = (height + text_height) // 2

    # Draw the message on the frame
    cv2.putText(frame, message, (x, y), font, font_scale, color, thickness)

    return frame
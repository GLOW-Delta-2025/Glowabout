# Serial Communication Settings
SERIAL_PORT = "COM4"
BAUD_RATE = 115200

# Video Settings
RTMP_URL = 0# "rtmp://192.168.1.188/live/live"
FRAME_SKIP = 2  # Skip every nth frame for processing

# Tracking Parameters
MOVEMENT_THRESHOLD = 10  # Minimum pixels of movement to register direction
MOMENTUM_FRAMES = 5     # Number of frames to maintain movement direction
IOU_THRESHOLD = 0.5     # Threshold for filtering overlapping detections

# Detection Parameters
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for person detection
PERSON_CLASS_ID = 0        # YOLO class ID for person

# Visualization Settings
BOUNDING_BOX_THICKNESS = 3
ARROW_LENGTH = 30
ARROW_THICKNESS = 3
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLACK = (0, 0, 0)

# Window Settings
WINDOW_NAME = "Webcam - Person Tracking"

# Message Format
MESSAGE_START = '$'
MESSAGE_END = '#'
MESSAGE_SEPARATOR = ':' 
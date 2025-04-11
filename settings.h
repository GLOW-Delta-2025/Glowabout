#ifndef SETTINGS_H
#define SETTINGS_H

// Serial Communication Settings
#define SERIAL_PORT "COM4"
#define BAUD_RATE 9600

// Video Settings
#define RTMP_URL "rtmp://192.168.1.188/live/live"
#define FRAME_SKIP 2  // Skip every nth frame for processing

// Tracking Parameters
#define MOVEMENT_THRESHOLD 10  // Minimum pixels of movement to register direction
#define MOMENTUM_FRAMES 5     // Number of frames to maintain movement direction
#define IOU_THRESHOLD 0.5     // Threshold for filtering overlapping detections

// Detection Parameters
#define CONFIDENCE_THRESHOLD 0.5  // Minimum confidence for person detection
#define PERSON_CLASS_ID 0        // YOLO class ID for person

// Visualization Settings
#define BOUNDING_BOX_THICKNESS 3
#define ARROW_LENGTH 30
#define ARROW_THICKNESS 3
#define TEXT_SCALE 0.7
#define TEXT_THICKNESS 2

// Colors (BGR format)
#define COLOR_GREEN (0, 255, 0)
#define COLOR_RED (0, 0, 255)
#define COLOR_BLACK (0, 0, 0)

// Window Settings
#define WINDOW_NAME "Webcam - Person Tracking"

// Message Format
#define MESSAGE_START '$'
#define MESSAGE_END '#'
#define MESSAGE_SEPARATOR ':'

#endif // SETTINGS_H 
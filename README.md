# Glowabout Project

This project appears to be a computer vision application using YOLOv8 for object detection, with additional functionality for circle detection and processing.

## Project Structure

- `Circle-two.py`: Main application file
- `settings.py` and `settings.h`: Configuration files
- YOLO model files:
  - `yolov8n.pt`: YOLOv8 nano model
  - `yolov8s.pt`: YOLOv8 small model
  - `yolov3.cfg` and `yolov3.weights`: YOLOv3 model files
- `coco.names`: COCO dataset class names

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`:
  - numpy==2.1.1
  - opencv-python==4.11.0.86
  - pyserial==3.5
  - torch==2.6.0
  - torchvision==0.21.0
  - ultralytics==8.3.93
  - ultralytics-thop==2.0.14

## Setup Instructions

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure all model files are in the correct location
2. Run the main application:
   ```bash
   python Circle-two.py
   ```

## Notes

- The project uses YOLOv8 for object detection
- Make sure you have sufficient disk space for the model files
- GPU acceleration is recommended for better performance
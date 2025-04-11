import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
import serial
from typing import Dict, List, Tuple, Optional
import settings  # Import settings

class PersonTracker:
    def __init__(self, com_port: str = settings.SERIAL_PORT, baud_rate: int = settings.BAUD_RATE):
        """Initialize the PersonTracker with model and tracking parameters."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize serial communication
        self.ser = self._initialize_serial(com_port, baud_rate)
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        self.model.to(self.device)
        
        # Tracking parameters
        self.movement_threshold = settings.MOVEMENT_THRESHOLD
        self.momentum_frames = settings.MOMENTUM_FRAMES
        self.frame_skip = settings.FRAME_SKIP
        
        # Tracking state
        self.tracking_history: Dict = {}
        self.last_detected_boxes: List = []
        self.last_centroids: List = []
        self.momentum_counter: Dict = {}
        self.last_printed_count = -1
        self.last_movement_states: Dict = {}
        self.frame_counter = 0
        
        # Video capture
        self.cap = None
        self.window_name = settings.WINDOW_NAME
        
        # Timing for serial communication
        self.last_serial_time = 0
        self.serial_interval = 0.5  # Send data every 1 second
        
        # Averaging variables
        self.left_counts = []
        self.right_counts = []
        self.total_counts = []
        self.measurement_times = []
        
    def _initialize_serial(self, com_port: str, baud_rate: int) -> Optional[serial.Serial]:
        """Initialize serial communication."""
        try:
            ser = serial.Serial(com_port, baud_rate)
            print("Serial connection established")
            return ser
        except serial.SerialException as e:
            print(f"Error: Could not open serial port: {e}")
            return None

    def _calculate_iou(self, boxA: Tuple, boxB: Tuple) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        return interArea / float(boxAArea + boxBArea - interArea)

    def _process_detections(self, results, frame_width: int, frame_height: int) -> Tuple[List, List]:
        """Process YOLO detections and return filtered boxes and centroids."""
        boxes = []
        centroids = []

        for result in results:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box_xyxy, conf, cls in zip(boxes_xyxy, confidences, class_ids):
                if conf > settings.CONFIDENCE_THRESHOLD and int(cls) == settings.PERSON_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box_xyxy)
                    x1, y1 = max(0, min(x1, frame_width)), max(0, min(y1, frame_height))
                    x2, y2 = max(0, min(x2, frame_width)), max(0, min(y2, frame_height))
                    
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
                    centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))

        # Filter overlapping boxes
        filtered_boxes = []
        filtered_centroids = []
        for box, centroid in zip(boxes, centroids):
            if not any(self._calculate_iou(box, other_box) > settings.IOU_THRESHOLD for other_box in filtered_boxes):
                filtered_boxes.append(box)
                filtered_centroids.append(centroid)

        return filtered_boxes, filtered_centroids

    def _determine_movement(self, current_x: int, prev_x: int) -> str:
        """Determine movement direction based on position change."""
        if abs(current_x - prev_x) > self.movement_threshold:
            return "Right" if current_x > prev_x else "Left"
        return "Staying"

    def _update_tracking(self, filtered_boxes: List, filtered_centroids: List) -> None:
        """Update tracking information for all detected persons."""
        current_tracks = {}
        
        for i, (box, centroid) in enumerate(zip(filtered_boxes, filtered_centroids)):
            current_x = centroid[0]
            
            # Find closest previous centroid
            min_dist = float('inf')
            closest_prev_idx = None
            for prev_idx, prev_centroid in enumerate(self.last_centroids):
                dist = abs(prev_centroid[0] - current_x)
                if dist < min_dist:
                    min_dist = dist
                    closest_prev_idx = prev_idx

            # Determine movement
            current_movement = "Staying"
            if closest_prev_idx is not None:
                prev_x = self.last_centroids[closest_prev_idx][0]
                current_movement = self._determine_movement(current_x, prev_x)

            # Update momentum
            if i not in self.momentum_counter:
                self.momentum_counter[i] = 0
            
            prev_movement = self.tracking_history.get(i, {}).get('movement', "Staying")
            
            # Apply momentum
            if current_movement != "Staying":
                movement = current_movement
                self.momentum_counter[i] = self.momentum_frames
            else:
                if self.momentum_counter[i] > 0:
                    movement = prev_movement
                    self.momentum_counter[i] -= 1
                else:
                    movement = "Staying"

            current_tracks[i] = {
                'box': box,
                'centroid': centroid,
                'movement': movement
            }

        self.tracking_history = current_tracks
        self.last_detected_boxes = filtered_boxes
        self.last_centroids = filtered_centroids

    def _send_movement_data(self, left_count: int, right_count: int, total_count: int) -> None:
        """Send movement data via serial communication."""
        if self.ser is not None and self.ser.is_open:
            try:
                # Calculate averages
                avg_left = int(np.mean(self.left_counts)) if self.left_counts else 0
                avg_right = int(np.mean(self.right_counts)) if self.right_counts else 0
                avg_total = int(np.mean(self.total_counts)) if self.total_counts else 0
                
                # Calculate average time between measurements
                if len(self.measurement_times) > 1:
                    time_diffs = np.diff(self.measurement_times)
                    avg_time = np.mean(time_diffs)
                else:
                    avg_time = 0
                
                # Send both current and average values
                message = f"{settings.MESSAGE_START}{avg_left}{settings.MESSAGE_SEPARATOR}{avg_right}{settings.MESSAGE_END}"
                print(f"Current: Left={left_count}, Right={right_count}, Total={total_count}")
                print(f"Average: Left={avg_left}, Right={avg_right}, Total={avg_total}, Time={avg_time:.2f}s")
                self.ser.write(message.encode())
                
                # Reset averaging arrays
                self.left_counts = []
                self.right_counts = []
                self.total_counts = []
                self.measurement_times = []
                
            except serial.SerialException as e:
                print(f"Error sending serial data: {e}")

    def _draw_tracking_info(self, frame: np.ndarray) -> None:
        """Draw tracking information on the frame."""
        for track_id, track_info in self.tracking_history.items():
            x, y, w, h = track_info['box']
            movement = track_info['movement']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), settings.COLOR_GREEN, settings.BOUNDING_BOX_THICKNESS)
            
            # Draw label background
            cv2.rectangle(frame, (x, y - 60), (x + 150, y), settings.COLOR_GREEN, -1)
            
            # Draw labels
            cv2.putText(frame, f"Person {track_id + 1}", (x + 5, y - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, settings.TEXT_SCALE, settings.COLOR_BLACK, settings.TEXT_THICKNESS)
            cv2.putText(frame, f"Moving: {movement}", (x + 5, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, settings.TEXT_SCALE, settings.COLOR_BLACK, settings.TEXT_THICKNESS)

            # Draw movement arrow
            if movement != "Staying":
                start_point = (x + w//2, y + h//2)
                end_point = (start_point[0] + settings.ARROW_LENGTH if movement == "Right" 
                           else start_point[0] - settings.ARROW_LENGTH, start_point[1])
                cv2.arrowedLine(frame, start_point, end_point, settings.COLOR_GREEN, settings.ARROW_THICKNESS)

    def _draw_stats(self, frame: np.ndarray, processing_time: float, fps: float) -> None:
        """Draw statistics on the frame."""
        moving_count = sum(1 for track_info in self.tracking_history.values() 
                         if track_info['movement'] != "Staying")
        left_count = sum(1 for movement in self.tracking_history.values() 
                       if movement['movement'] == "Left")
        right_count = sum(1 for movement in self.tracking_history.values() 
                        if movement['movement'] == "Right")

        # Draw processing info
        cv2.putText(frame, f'Processing Time: {processing_time:.3f}s', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, settings.TEXT_SCALE, settings.COLOR_RED, settings.TEXT_THICKNESS)
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, settings.TEXT_SCALE, settings.COLOR_RED, settings.TEXT_THICKNESS)
        cv2.putText(frame, f'Device: {self.device}', (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, settings.TEXT_SCALE, settings.COLOR_RED, settings.TEXT_THICKNESS)

        # Draw counts
        cv2.putText(frame, f'Current Count: {len(self.tracking_history)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, settings.COLOR_RED, settings.TEXT_THICKNESS)
        cv2.putText(frame, f'Moving: {moving_count}', (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, settings.COLOR_RED, settings.TEXT_THICKNESS)
        cv2.putText(frame, f'Left: {left_count} - Right: {right_count}', (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, settings.COLOR_RED, settings.TEXT_THICKNESS)

        return moving_count, left_count, right_count

    def start_tracking(self, rtmp_url: str = settings.RTMP_URL) -> None:
        """Start the person tracking process."""
        self.cap = cv2.VideoCapture(rtmp_url)
        
        # Get frame dimensions
        ret, first_frame = self.cap.read()
        if not ret:
            print("Error: Could not read first frame.")
            return
            
        frame_height, frame_width = first_frame.shape[:2]
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, frame_width, frame_height)

        while True:
            start_time = time.time()
            current_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            if self.frame_counter % self.frame_skip != 0:
                self.frame_counter += 1
                continue

            # Process frame
            results = self.model(frame, conf=settings.CONFIDENCE_THRESHOLD, verbose=False, device=self.device)
            filtered_boxes, filtered_centroids = self._process_detections(results, frame_width, frame_height)
            
            # Update tracking
            self._update_tracking(filtered_boxes, filtered_centroids)
            
            # Calculate and display stats
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time
            moving_count, left_count, right_count = self._draw_stats(frame, processing_time, fps)
            
            # Store measurements for averaging
            self.left_counts.append(left_count)
            self.right_counts.append(right_count)
            self.total_counts.append(moving_count)
            self.measurement_times.append(current_time)
            
            # Send data every second
            if current_time - self.last_serial_time >= self.serial_interval:
                self._send_movement_data(left_count, right_count, moving_count)
                self.last_serial_time = current_time

            # Draw tracking information
            self._draw_tracking_info(frame)
            
            # Show frame
            cv2.imshow(self.window_name, frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.frame_counter += 1

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed")

def main():
    tracker = PersonTracker()
    try:
        tracker.start_tracking()
    finally:
        tracker.cleanup()

if __name__ == "__main__":
    main()

"""
Vehicle Detection Module
Uses Ultralytics YOLO models: YOLOv8, YOLOv11 (official Ultralytics)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict


class VehicleDetector:
    """Vehicle detection using YOLO models"""
    
    def __init__(self, model_name: str = 'yolo11n', model_type: str = 'yolo11'):
        """
        Initialize vehicle detector with Ultralytics YOLO
        
        Args:
            model_name: Ultralytics YOLO model variant
                       - YOLOv8: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
                       - YOLO11: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
            model_type: Type of model ('yolov8', 'yolo11') - official Ultralytics models
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
        
        # Vehicle classes in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
    def load_model(self):
        """Load Ultralytics YOLO model"""
        try:
            # All models use Ultralytics YOLO API
            # YOLOv8 and YOLO11 are official Ultralytics models
            # Model name format: yolo11n.pt, yolov8n.pt, etc.
            self.model = YOLO(f'{self.model_name}.pt')
            
            print(f"✅ Loaded Ultralytics {self.model_type} model: {self.model_name} on {self.device}")
        except Exception as e:
            print(f"❌ Error loading Ultralytics YOLO model: {e}")
            raise
    
    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = 0.25, imgsz: int = 640) -> List[Dict]:
        """
        Detect vehicles in a frame
        
        Args:
            frame: Input image frame (BGR format)
            conf_threshold: Confidence threshold for detection
            imgsz: Input image size (smaller = faster, default 640)
            
        Returns:
            List of detected vehicles with bounding boxes and confidence scores
        """
        # Use smaller image size for faster inference (416 is faster than 640)
        results = self.model(frame, conf=conf_threshold, classes=self.vehicle_classes, 
                           imgsz=imgsz, verbose=False, half=False)  # half=False for CPU compatibility
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names.get(class_id, 'vehicle')
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn bounding boxes
        """
        output_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(output_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return output_frame
    
def create_detector(model_type: str = 'yolo11', model_size: str = 'n') -> VehicleDetector:
    """
    Factory function to create Ultralytics YOLO detector
    
    Args:
        model_type: 'yolov8' or 'yolo11' (official Ultralytics models)
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
    
    Returns:
        VehicleDetector instance using Ultralytics YOLO
    """
    model_name = f'{model_type}{model_size}'
    return VehicleDetector(model_name=model_name, model_type=model_type)


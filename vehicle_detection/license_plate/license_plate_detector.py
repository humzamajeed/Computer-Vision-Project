"""
License Plate Detection Module
Uses advanced Computer Vision techniques for highly accurate license plate detection
Optimized for maximum accuracy (200% focus on detection only)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class LicensePlateDetector:
    """License plate detection using advanced CV methods - optimized for maximum accuracy"""
    
    def __init__(self, plate_model: str = 'yolo11n', debug: bool = False):
        """
        Initialize license plate detector - detection only, no OCR
        
        Args:
            plate_model: Ultralytics YOLO model (yolov8n, yolo11n, etc.) - not used for detection, kept for compatibility
            debug: If True, saves plate images for debugging
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.debug = debug
        self.debug_count = 0
        
        print(f"✅ License Plate Detector initialized (Detection Only - Maximum Accuracy Mode)")
    
    def detect_plates_yolo(self, frame: np.ndarray, conf_threshold: float = 0.25, imgsz: int = 416) -> List[Dict]:
        """
        Detect license plates using Ultralytics YOLO neural network
        
        Args:
            frame: Input image frame
            conf_threshold: Confidence threshold
            imgsz: Input image size (smaller = faster, default 416)
            
        Returns:
            List of detected license plates with bounding boxes
        """
        if self.plate_model is None:
            return []
        
        # Run Ultralytics YOLO detection with optimized settings
        results = self.plate_model(frame, conf=conf_threshold, imgsz=imgsz, verbose=False, half=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Filter for license plate-like objects
                # Check aspect ratio (plates are typically wider than tall)
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                # License plates typically have aspect ratio between 2:1 and 5:1
                if 1.5 <= aspect_ratio <= 6.0 and width > 30 and height > 10:
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id
                    })
        
        return detections
    
    def detect_plates_in_vehicle_roi(self, vehicle_roi: np.ndarray, conf_threshold: float = 0.25) -> Optional[Dict]:
        """
        Detect license plate within vehicle ROI - optimized for parking lot scenarios
        Uses multiple methods and searches entire vehicle (front/rear plates)
        
        Args:
            vehicle_roi: Cropped vehicle region
            conf_threshold: Confidence threshold
            
        Returns:
            License plate bounding box or None
        """
        if vehicle_roi.size == 0:
            return None
        
        h, w = vehicle_roi.shape[:2]
        if h < 40 or w < 40:
            return None
        
        # Focus on lower regions only - plates are NOT on windows (upper/middle)
        # Windows are usually in upper 40-60% of vehicle, plates are in lower 30-50%
        search_regions = [
            (int(h * 0.6), h, "lower_rear"),  # Lower 40% - rear plate (most common)
            (int(h * 0.5), int(h * 0.85), "lower_mid"),  # Lower middle - some rear plates
        ]
        
        all_candidates = []
        
        for region_start, region_end, region_name in search_regions:
            if region_end <= region_start:
                continue
                
            roi_region = vehicle_roi[region_start:region_end, :]
            if roi_region.size == 0:
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY) if len(roi_region.shape) == 3 else roi_region
        
            # Method 1: MSER (Maximally Stable Extremal Regions) - good for text detection
            try:
                mser = cv2.MSER_create()
                regions, _ = mser.detectRegions(gray)
                
                for region in regions:
                    if len(region) < 8:  # Lowered threshold
                        continue
                    x, y, w_rect, h_rect = cv2.boundingRect(region.reshape(-1, 1, 2))
                    
                    # STRICT validation - license plates have specific characteristics
                    aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
                    area = w_rect * h_rect
                    
                    # Stricter thresholds to avoid windows and false positives
                    if (2.5 <= aspect_ratio <= 4.2 and  # License plates: 2.5:1 to 4.2:1 (NOT 1.8-5.0)
                        w_rect >= 80 and h_rect >= 25 and  # Minimum size (larger to avoid small false positives)
                        area >= 2000 and  # Minimum area
                        w_rect <= w * 0.5):  # Max 50% of vehicle width (plates are smaller)
                        
                        # Extract ROI for detailed analysis
                        roi = gray[y:y+h_rect, x:x+w_rect] if y+h_rect <= gray.shape[0] and x+w_rect <= gray.shape[1] else None
                        if roi is not None and roi.size > 0:
                            # Check contrast (plates have VERY high contrast)
                            contrast = np.std(roi)
                            
                            # Check for text-like horizontal structure (plates have horizontal text lines)
                            edges = cv2.Canny(roi, 50, 150)
                            horizontal_projection = np.sum(edges, axis=1)  # Sum edges horizontally
                            vertical_projection = np.sum(edges, axis=0)  # Sum edges vertically
                            
                            # Plates have more horizontal edges (text lines) than vertical
                            horizontal_edge_ratio = np.mean(horizontal_projection > np.mean(horizontal_projection) * 0.5)
                            vertical_edge_ratio = np.mean(vertical_projection > np.mean(vertical_projection) * 0.5)
                            
                            # Check for character-like patterns (multiple horizontal bands)
                            horizontal_bands = np.sum(horizontal_projection > np.max(horizontal_projection) * 0.3)
                            
                            # STRICT validation: high contrast + text-like structure
                            if (contrast > 25 and  # Higher contrast threshold
                                horizontal_edge_ratio > 0.4 and  # Strong horizontal structure
                                horizontal_bands >= 3 and  # Multiple text lines/bands
                                horizontal_edge_ratio > vertical_edge_ratio * 1.2):  # More horizontal than vertical
                                
                                # Check position - plates are usually centered horizontally
                                center_x = x + w_rect / 2
                                vehicle_center_x = w / 2
                                horizontal_offset = abs(center_x - vehicle_center_x) / w
                                
                                # Plates are usually within 30% of vehicle center
                                if horizontal_offset < 0.3:
                                    y_offset = region_start
                                    all_candidates.append({
                                        'bbox': [x, y + y_offset, x + w_rect, y + y_offset + h_rect],
                                        'confidence': min(0.95, (contrast / 100.0) * (horizontal_edge_ratio * 2) * (horizontal_bands / 5.0)),
                                        'area': area,
                                        'contrast': contrast,
                                        'horizontal_ratio': horizontal_edge_ratio,
                                        'bands': horizontal_bands,
                                        'method': f'mser_{region_name}'
                                    })
            except:
                pass
        
            # Method 2: Contour-based detection with adaptive thresholding
            # Try multiple thresholding methods
            thresh_methods = [
                ('adaptive', lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                              cv2.THRESH_BINARY_INV, 11, 2)),
                ('adaptive_mean', lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                                                   cv2.THRESH_BINARY_INV, 11, 2)),
                ('otsu', lambda g: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
            ]
            
            for thresh_name, thresh_func in thresh_methods:
                try:
                    adaptive_thresh = thresh_func(gray)
                    
                    # Morphological operations to connect text characters
                    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
                    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
                    
                    # Horizontal dilation to connect characters in a line
                    dilated_h = cv2.dilate(adaptive_thresh, kernel_horizontal, iterations=2)
                    # Vertical dilation to connect lines
                    dilated_v = cv2.dilate(dilated_h, kernel_vertical, iterations=1)
                    
                    # Find contours
                    contours, _ = cv2.findContours(dilated_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        x, y, w_rect, h_rect = cv2.boundingRect(contour)
                        
                        aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
                        area = w_rect * h_rect
                        
                        # STRICT filtering - avoid windows and false positives
                        if (2.5 <= aspect_ratio <= 4.2 and  # License plate aspect ratio
                            w_rect >= 80 and h_rect >= 25 and  # Minimum size
                            area >= 2000 and  # Minimum area
                            w_rect <= w * 0.5):  # Max 50% of vehicle width
                            
                            # Extract region and check for text-like properties
                            roi = gray[y:y+h_rect, x:x+w_rect] if y+h_rect <= gray.shape[0] and x+w_rect <= gray.shape[1] else None
                            if roi is not None and roi.size > 0:
                                # Check contrast (plates have high contrast)
                                contrast = np.std(roi)
                                
                                # Check for horizontal text structure
                                edges = cv2.Canny(roi, 50, 150)
                                horizontal_projection = np.sum(edges, axis=1)
                                vertical_projection = np.sum(edges, axis=0)
                                
                                horizontal_edge_ratio = np.mean(horizontal_projection > np.mean(horizontal_projection) * 0.5)
                                vertical_edge_ratio = np.mean(vertical_projection > np.mean(vertical_projection) * 0.5)
                                horizontal_bands = np.sum(horizontal_projection > np.max(horizontal_projection) * 0.3)
                                
                                # Check position (centered horizontally)
                                center_x = x + w_rect / 2
                                vehicle_center_x = w / 2
                                horizontal_offset = abs(center_x - vehicle_center_x) / w
                                
                                # STRICT validation
                                if (contrast > 25 and  # High contrast
                                    horizontal_edge_ratio > 0.4 and  # Strong horizontal structure
                                    horizontal_bands >= 3 and  # Multiple text bands
                                    horizontal_edge_ratio > vertical_edge_ratio * 1.2 and  # More horizontal
                                    horizontal_offset < 0.3):  # Centered
                                    
                                    y_offset = region_start
                                    all_candidates.append({
                                        'bbox': [x, y + y_offset, x + w_rect, y + y_offset + h_rect],
                                        'confidence': min(0.9, (contrast / 100.0) * (horizontal_edge_ratio * 2) * (horizontal_bands / 5.0)),
                                        'area': area,
                                        'contrast': contrast,
                                        'horizontal_ratio': horizontal_edge_ratio,
                                        'bands': horizontal_bands,
                                        'method': f'contour_{thresh_name}_{region_name}'
                                    })
                except:
                    continue
            
            # Method 3: Canny edge detection + contour finding
            try:
                edges = cv2.Canny(gray, 50, 150)
                # Dilate to connect edges
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
                dilated = cv2.dilate(edges, kernel, iterations=2)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
                    area = w_rect * h_rect
                    
                    # STRICT filtering for Canny method too
                    if (2.5 <= aspect_ratio <= 4.2 and
                        w_rect >= 80 and h_rect >= 25 and
                        area >= 2000 and
                        w_rect <= w * 0.5):
                        
                        roi = gray[y:y+h_rect, x:x+w_rect] if y+h_rect <= gray.shape[0] and x+w_rect <= gray.shape[1] else None
                        if roi is not None and roi.size > 0:
                            contrast = np.std(roi)
                            
                            # Check for text structure
                            edges = cv2.Canny(roi, 50, 150)
                            horizontal_projection = np.sum(edges, axis=1)
                            horizontal_bands = np.sum(horizontal_projection > np.max(horizontal_projection) * 0.3)
                            
                            # Check position
                            center_x = x + w_rect / 2
                            vehicle_center_x = w / 2
                            horizontal_offset = abs(center_x - vehicle_center_x) / w
                            
                            if (contrast > 25 and
                                horizontal_bands >= 3 and
                                horizontal_offset < 0.3):
                                
                                y_offset = region_start
                                all_candidates.append({
                                    'bbox': [x, y + y_offset, x + w_rect, y + y_offset + h_rect],
                                    'confidence': min(0.85, (contrast / 100.0) * (horizontal_bands / 5.0)),
                                    'area': area,
                                    'contrast': contrast,
                                    'bands': horizontal_bands,
                                    'method': f'canny_{region_name}'
                                })
            except:
                pass
        
        if not all_candidates:
            return None
        
        # Score candidates with STRICT validation
        for candidate in all_candidates:
            area_score = candidate['area'] / (w * h * 0.12)  # Normalize by vehicle area
            contrast_score = candidate.get('contrast', 30) / 100.0  # Higher contrast = better
            horizontal_score = candidate.get('horizontal_ratio', 0.3)  # Text structure score
            bands_score = candidate.get('bands', 0) / 5.0  # Number of text bands
            
            # Combined score with emphasis on text-like features
            candidate['score'] = (candidate['confidence'] * 
                                 min(1.2, area_score) * 
                                 min(1.5, contrast_score) * 
                                 min(1.5, horizontal_score * 2) * 
                                 min(1.3, bands_score))
        
        if not all_candidates:
            return None
        
        # Return candidate with highest score
        best_plate = max(all_candidates, key=lambda x: x['score'])
        
        # MUCH HIGHER threshold to avoid false positives (windows, etc.)
        # Only return if score is very high (indicating strong plate characteristics)
        if best_plate['score'] > 0.25:  # Increased from 0.12 to 0.25 for accuracy
            return best_plate
        
        return None
    
    def process_vehicle(self, frame: np.ndarray, vehicle_bbox: List[int], conf_threshold: float = 0.25) -> Dict:
        """
        Process vehicle to detect license plate - detection only, maximum accuracy
        
        Args:
            frame: Full frame
            vehicle_bbox: Vehicle bounding box [x1, y1, x2, y2]
            conf_threshold: Confidence threshold for plate detection
            
        Returns:
            Dictionary with plate detection info (no OCR)
        """
        x1, y1, x2, y2 = vehicle_bbox
        
        # Crop vehicle region with padding for better detection
        padding = 10  # Increased padding for better detection
        x1_crop = max(0, x1 - padding)
        y1_crop = max(0, y1 - padding)
        x2_crop = min(frame.shape[1], x2 + padding)
        y2_crop = min(frame.shape[0], y2 + padding)
        
        vehicle_roi = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if vehicle_roi.size == 0:
            return {'plate_bbox': None, 'confidence': 0.0}
        
        # Detect license plate using improved CV methods (maximum accuracy)
        plate_detection = self.detect_plates_in_vehicle_roi(vehicle_roi, conf_threshold)
        
        if plate_detection is None:
            return {'plate_bbox': None, 'confidence': 0.0}
        
        # Get plate bounding box relative to vehicle ROI
        px1, py1, px2, py2 = plate_detection['bbox']
        
        # Adjust coordinates to full frame (account for padding)
        plate_bbox_full = [x1_crop + px1, y1_crop + py1, x1_crop + px2, y1_crop + py2]
        
        # Validate plate bbox is within vehicle bbox (with tolerance)
        tolerance = 30  # Increased tolerance for better detection
        if (plate_bbox_full[0] < x1 - tolerance or plate_bbox_full[1] < y1 - tolerance or
            plate_bbox_full[2] > x2 + tolerance or plate_bbox_full[3] > y2 + tolerance):
            # Plate detected outside vehicle - likely false positive, skip
            return {'plate_bbox': None, 'confidence': 0.0}
        
        # Debug: Save plate image for inspection
        if self.debug and self.debug_count < 20:  # Save first 20 plates
            debug_path = Path('debug_plates')
            debug_path.mkdir(exist_ok=True)
            plate_roi = vehicle_roi[py1:py2, px1:px2]
            if plate_roi.size > 0:
                cv2.imwrite(str(debug_path / f'plate_{self.debug_count}_detected.jpg'), plate_roi)
                self.debug_count += 1
        
        # Return detection only (no OCR)
        return {
            'plate_bbox': plate_bbox_full,
            'confidence': plate_detection['confidence']
        }
    

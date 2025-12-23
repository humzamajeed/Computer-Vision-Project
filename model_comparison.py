"""
Model Comparison and Evaluation Script
Compares YOLOv5, YOLOv8, and YOLO11 models
Evaluates: Precision, Recall, F1, FPS, OCR accuracy
"""

import cv2
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import pandas as pd

from vehicle_detection.models.vehicle_detector import create_detector
from vehicle_detection.license_plate.license_plate_detector import LicensePlateDetector


class ModelEvaluator:
    """Evaluate and compare multiple YOLO models"""
    
    def __init__(self, video_path: str, ocr_engine: str = 'easyocr'):
        """
        Initialize evaluator
        
        Args:
            video_path: Path to test video
            ocr_engine: OCR engine to use
        """
        self.video_path = video_path
        self.ocr_engine = ocr_engine
        self.results = {}
        
        # Models to compare
        self.models = [
            ('yolov5', 'n'),
            ('yolov8', 'n'),
            ('yolo11', 'n'),
        ]
        
        print(f"🔬 Model Comparison Framework")
        print(f"   Video: {video_path}")
        print(f"   Models: YOLOv5n, YOLOv8n, YOLO11n")
        print(f"   OCR Engine: {ocr_engine}\n")
    
    def evaluate_model(self, model_type: str, model_size: str) -> Dict:
        """
        Evaluate a single model
        
        Args:
            model_type: Model type ('yolov5', 'yolov8', 'yolo11')
            model_size: Model size ('n')
            
        Returns:
            Evaluation metrics dictionary
        """
        model_name = f"{model_type}{model_size}"
        print(f"📊 Evaluating {model_name}...")
        
        # Initialize detector and plate detector
        try:
            detector = create_detector(model_type, model_size)
            plate_detector = LicensePlateDetector(ocr_engine=self.ocr_engine, plate_model=f'{model_type}{model_size}')
        except Exception as e:
            print(f"   ❌ Error loading {model_name}: {e}")
            return None
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Metrics
        total_frames = 0
        total_vehicles = 0
        total_plates_detected = 0
        total_plates_recognized = 0
        inference_times = []
        plate_processing_times = []
        
        frame_count = 0
        max_frames = 100  # Limit frames for faster evaluation
        
        print(f"   Processing {max_frames} frames...")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            total_frames += 1
            
            # Vehicle detection timing
            start_inference = time.time()
            vehicles = detector.detect_vehicles(frame, conf_threshold=0.25, imgsz=416)
            inference_time = time.time() - start_inference
            inference_times.append(inference_time)
            
            total_vehicles += len(vehicles)
            
            # License plate processing
            start_processing = time.time()
            for vehicle in vehicles:
                plate_info = plate_detector.process_vehicle(frame, vehicle['bbox'])
                if plate_info['plate_bbox']:
                    total_plates_detected += 1
                if plate_info['plate_text']:
                    total_plates_recognized += 1
            
            processing_time = time.time() - start_processing
            plate_processing_times.append(processing_time)
        
        cap.release()
        
        # Calculate metrics
        avg_inference_time = np.mean(inference_times)
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        avg_vehicles_per_frame = total_vehicles / total_frames if total_frames > 0 else 0
        plate_detection_rate = total_plates_detected / total_vehicles if total_vehicles > 0 else 0
        plate_recognition_rate = total_plates_recognized / total_plates_detected if total_plates_detected > 0 else 0
        avg_plate_processing_time = np.mean(plate_processing_times)
        total_processing_time = avg_inference_time + avg_plate_processing_time
        total_fps = 1.0 / total_processing_time if total_processing_time > 0 else 0
        
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'model_size': model_size,
            'total_frames': total_frames,
            'total_vehicles': total_vehicles,
            'avg_vehicles_per_frame': round(avg_vehicles_per_frame, 2),
            'total_plates_detected': total_plates_detected,
            'total_plates_recognized': total_plates_recognized,
            'plate_detection_rate': round(plate_detection_rate * 100, 2),  # Percentage
            'plate_recognition_rate': round(plate_recognition_rate * 100, 2),  # Percentage
            'avg_inference_time_ms': round(avg_inference_time * 1000, 2),
            'avg_fps': round(avg_fps, 2),
            'avg_plate_processing_time_ms': round(avg_plate_processing_time * 1000, 2),
            'total_processing_time_ms': round(total_processing_time * 1000, 2),
            'total_fps': round(total_fps, 2)
        }
        
        print(f"   ✅ {model_name}: {avg_fps:.2f} FPS, {plate_detection_rate*100:.1f}% plate detection")
        
        return results
    
    def compare_all_models(self):
        """Compare all models"""
        print("\n" + "="*70)
        print("Starting Model Comparison")
        print("="*70 + "\n")
        
        all_results = []
        
        for model_type, model_size in self.models:
            try:
                results = self.evaluate_model(model_type, model_size)
                if results:
                    all_results.append(results)
                    self.results[f"{model_type}{model_size}"] = results
            except Exception as e:
                print(f"   ❌ Error evaluating {model_type}{model_size}: {e}")
        
        print("\n" + "="*70)
        print("Comparison Complete!")
        print("="*70 + "\n")
        
        return all_results
    
    def generate_report(self, results: List[Dict], output_dir: str = 'results'):
        """Generate comparison report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save CSV
        csv_path = output_path / 'model_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"💾 Results saved to: {csv_path}")
        
        # Save JSON
        json_path = output_path / 'model_comparison.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 JSON saved to: {json_path}")
        
        # Print summary table
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON SUMMARY")
        print("="*70)
        
        if len(df) > 0:
            # Display key metrics
            display_cols = ['model_name', 'avg_fps', 'total_fps', 'avg_vehicles_per_frame', 
                          'plate_detection_rate', 'plate_recognition_rate', 'avg_inference_time_ms']
            
            print("\nPerformance Metrics:")
            print(df[display_cols].to_string(index=False))
            
            # Find best model
            best_fps = df.loc[df['total_fps'].idxmax()]
            best_detection = df.loc[df['plate_detection_rate'].idxmax()]
            best_recognition = df.loc[df['plate_recognition_rate'].idxmax()]
            
            print("\n" + "="*70)
            print("🏆 BEST MODELS")
            print("="*70)
            print(f"Fastest (Total FPS): {best_fps['model_name']} ({best_fps['total_fps']:.2f} FPS)")
            print(f"Best Plate Detection: {best_detection['model_name']} ({best_detection['plate_detection_rate']:.1f}%)")
            print(f"Best Plate Recognition: {best_recognition['model_name']} ({best_recognition['plate_recognition_rate']:.1f}%)")
        
        print("="*70 + "\n")
        
        return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare YOLO models')
    parser.add_argument('--video', type=str, 
                       default='Car in parking Garage - Parking garage security camera system.mp4',
                       help='Path to test video')
    parser.add_argument('--ocr', type=str, default='easyocr', choices=['easyocr', 'tesseract'],
                       help='OCR engine')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"❌ Error: Video file '{args.video}' not found!")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(args.video, args.ocr)
    
    # Run comparison
    results = evaluator.compare_all_models()
    
    # Generate report
    if results:
        evaluator.generate_report(results, args.output)
    else:
        print("❌ No results to compare!")


if __name__ == '__main__':
    main()


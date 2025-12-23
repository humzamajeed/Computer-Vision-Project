"""
Simple script to run vehicle and license plate detection on video
Usage: python run_detection.py
"""

import cv2
import sys
from pathlib import Path
from vehicle_detection.models.vehicle_detector import create_detector
from vehicle_detection.license_plate.license_plate_detector import LicensePlateDetector

def main():
    # Video file name
    video_file = "Car in parking Garage - Parking garage security camera system.mp4"
    
    # Check if video exists
    if not Path(video_file).exists():
        print(f"❌ Error: Video file '{video_file}' not found!")
        print("Please make sure the video file is in the current directory.")
        return
    
    print("🚀 Starting Vehicle and License Plate Detection (Detection Only)")
    print("=" * 60)
    print(f"📹 Video: {video_file}")
    print("🔍 Model: YOLO11 Nano (Ultralytics - Latest)")
    print("🎯 Mode: Detection Only - Maximum Accuracy")
    print("=" * 60)
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 'p' to pause/resume")
    print("  - Green boxes = Vehicles (shows: 'class_name: confidence_score')")
    print("  - Blue boxes = License Plates (shows: 'Plate Detected')")
    print("\n📝 Explanation:")
    print("   - 'car: 0.89' means: Vehicle class 'car' detected with 89% confidence")
    print("   - Confidence score (0.00-1.00) shows how sure the model is")
    print("   - Higher score = More confident detection")
    print("   - Blue box = License plate detected (no OCR)")
    print("\n")
    
    # Initialize detectors with Ultralytics YOLO11
    print("⏳ Loading Ultralytics YOLO11 models (this may take a minute on first run)...")
    print("   - Loading YOLO11 Nano (Ultralytics - Latest) for vehicle detection...")
    try:
        vehicle_detector = create_detector('yolo11', 'n')  # Ultralytics YOLO11
        print("   - Initializing license plate detector (Detection Only - Maximum Accuracy)...")
        # Detection only - no OCR, maximum accuracy focus
        plate_detector = LicensePlateDetector(plate_model='yolo11n', debug=True)
        print("✅ All Ultralytics YOLO11 models loaded successfully!\n")
        print("📌 Using Ultralytics YOLO11 (Latest) for both vehicle and license plate detection\n")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file!")
        return
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f} seconds\n")
    
    # Setup video writer to save processed video
    output_video_file = "output_detection.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))
    print(f"💾 Saving processed video to: {output_video_file}\n")
    
    frame_count = 0
    vehicle_count = 0
    plate_count = 0
    paused = False
    
    print("🎬 Processing video...\n")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Video processing complete!")
                break
            
            frame_count += 1
            
            # Detect vehicles with optimized settings for speed
            vehicles = vehicle_detector.detect_vehicles(frame, conf_threshold=0.25, imgsz=416)
            vehicle_count += len(vehicles)
            
            # Process each vehicle for license plate
            for vehicle in vehicles:
                # Draw vehicle bounding box (green, thicker)
                x1, y1, x2, y2 = vehicle['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add vehicle label with background
                # Format: "Class Name: Confidence Score"
                # Example: "car: 0.89" means "car" class detected with 89% confidence
                label = f"{vehicle['class_name']}: {vehicle['confidence']:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                             (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Detect license plate (detection only, no OCR)
                plate_info = plate_detector.process_vehicle(frame, vehicle['bbox'], conf_threshold=0.25)
                
                # Only show plate if it's actually detected (not None)
                if plate_info['plate_bbox'] is not None:
                    px1, py1, px2, py2 = plate_info['plate_bbox']
                    
                    # Validate plate bbox is reasonable (not too large, within vehicle)
                    vx1, vy1, vx2, vy2 = vehicle['bbox']
                    plate_w = px2 - px1
                    plate_h = py2 - py1
                    vehicle_w = vx2 - vx1
                    vehicle_h = vy2 - vy1
                    
                    # STRICT validation - plates are smaller and in lower portion
                    # Plates are usually 10-30% of vehicle width and 5-15% of vehicle height
                    plate_width_ratio = plate_w / vehicle_w
                    plate_height_ratio = plate_h / vehicle_h
                    plate_y_ratio = (py1 + py2) / 2 / (vy1 + vy2) * 2  # Position in vehicle (0=top, 1=bottom)
                    
                    # Plates should be:
                    # - 10-40% of vehicle width (not too large like windows)
                    # - 5-20% of vehicle height
                    # - In lower 60% of vehicle (not upper where windows are)
                    # - Minimum size: 60x20 pixels
                    if (0.1 <= plate_width_ratio <= 0.4 and  # 10-40% of vehicle width
                        0.05 <= plate_height_ratio <= 0.2 and  # 5-20% of vehicle height
                        plate_y_ratio > 0.4 and  # In lower 60% of vehicle (not upper windows)
                        plate_w > 60 and plate_h > 20 and  # Minimum size
                        px1 >= vx1 - 30 and py1 >= vy1 - 30 and
                        px2 <= vx2 + 30 and py2 <= vy2 + 30):
                        
                        # Draw license plate bounding box (blue, thicker) - should be smaller than vehicle box
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 3)
                        
                        # Add license plate label (detection only, no OCR)
                        plate_text = f"Plate: {plate_info['confidence']:.2f}"
                        plate_count += 1
                        
                        (text_width, text_height), baseline = cv2.getTextSize(
                            plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (px1, py1 - text_height - 10), 
                                     (px1 + text_width, py1), (255, 0, 0), -1)
                        cv2.putText(frame, plate_text, (px1, py1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add info overlay
            info_text = f"Frame: {frame_count}/{total_frames} | Vehicles: {len(vehicles)} | Plates: {plate_count}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Green = Vehicle | Blue = License Plate", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Display frame
        cv2.imshow('Vehicle & License Plate Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            status = "⏸️  PAUSED" if paused else "▶️  RESUMED"
            print(status)
    
    # Cleanup
    cap.release()
    out.release()  # Release video writer
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Detection Summary:")
    print(f"   Total Frames Processed: {frame_count}")
    print(f"   Total Vehicles Detected: {vehicle_count}")
    print(f"   Total License Plates Found: {plate_count}")
    print(f"   ✅ Processed video saved to: {output_video_file}")
    print("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


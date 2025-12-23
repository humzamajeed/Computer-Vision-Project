"""
Simple script to verify and view the saved detection video
Usage: python verify_video.py
"""

import cv2
import sys
from pathlib import Path

def verify_video(video_file="output_detection.mp4"):
    """Verify and display the saved video"""
    
    if not Path(video_file).exists():
        print(f"❌ Error: Video file '{video_file}' not found!")
        print("Please run 'python run_detection.py' first to generate the video.")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file '{video_file}'!")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print("=" * 60)
    print("📹 VIDEO VERIFICATION")
    print("=" * 60)
    print(f"📁 File: {video_file}")
    print(f"📊 Resolution: {width}x{height}")
    print(f"🎬 FPS: {fps:.2f}")
    print(f"🎞️  Total Frames: {total_frames}")
    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"💾 File Size: {Path(video_file).stat().st_size / (1024*1024):.2f} MB")
    print("=" * 60)
    print("\n✅ Video file is valid and readable!")
    print("\n🎬 Playing video...")
    print("   - Press 'q' to quit")
    print("   - Press SPACE to pause/resume")
    print("   - Press 'f' to toggle fullscreen")
    print("\n")
    
    frame_count = 0
    paused = False
    fullscreen = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Video playback complete!")
                break
            
            frame_count += 1
            
            # Add frame info overlay
            info_text = f"Frame: {frame_count}/{total_frames} | Time: {frame_count/fps:.1f}s"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit | SPACE to pause | 'f' for fullscreen", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        window_name = 'Video Verification - Press q to quit'
        if fullscreen:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        cv2.imshow(window_name, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar
            paused = not paused
            status = "⏸️  PAUSED" if paused else "▶️  RESUMED"
            print(status)
        elif key == ord('f'):
            fullscreen = not fullscreen
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("✅ Video verification complete!")
    print("=" * 60)

if __name__ == '__main__':
    try:
        # Check if custom video file provided
        video_file = sys.argv[1] if len(sys.argv) > 1 else "output_detection.mp4"
        verify_video(video_file)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


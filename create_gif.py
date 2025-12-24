"""
Create a slow animated GIF from output_detection.mp4 for README display
Optimized for minimal speed and GitHub file size limits
"""

import cv2
import imageio
from pathlib import Path
import sys
import io

# Fix for UnicodeEncodeError in some environments
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def create_gif_from_video(video_path: str, output_path: str, max_frames: int = 80, fps: float = 0.5, scale: float = 0.35):
    """
    Create a slow animated GIF from video
    
    Args:
        video_path: Path to input video file
        output_path: Path to output GIF file
        max_frames: Maximum number of frames to include
        fps: Frames per second for GIF (lower = slower)
        scale: Scale factor for resizing (0.35 = 35% of original size)
    """
    if not Path(video_path).exists():
        print(f"❌ Error: Video file '{video_path}' not found!")
        return False
    
    print(f"🎬 Creating slow GIF from: {video_path}")
    print(f"   Target: {output_path}")
    print(f"   Settings: {max_frames} frames, {fps} FPS, {scale*100:.0f}% scale")
    print("   This may take a minute...\n")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file!")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame step to get max_frames evenly distributed
    frame_step = max(1, total_frames // max_frames)
    
    frames = []
    frame_count = 0
    saved_count = 0
    
    print("📸 Extracting frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only save every Nth frame
        if frame_count % frame_step == 0 and saved_count < max_frames:
            # Resize frame
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert BGR to RGB for imageio
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"   Extracted {saved_count}/{max_frames} frames...")
        
        frame_count += 1
    
    cap.release()
    
    if not frames:
        print("❌ Error: No frames extracted!")
        return False
    
    print(f"\n💾 Saving GIF: {output_path}")
    print(f"   Frames: {len(frames)}")
    print(f"   FPS: {fps} (very slow for better visibility)")
    print(f"   Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
    print("   This may take a minute...")
    
    # Calculate duration per frame (in seconds) for very slow playback
    duration_per_frame = 1.0 / fps if fps > 0 else 2.0
    
    imageio.mimsave(
        output_path,
        frames,
        fps=fps,
        loop=0,  # Infinite loop
        duration=duration_per_frame  # Control frame duration for slow playback
    )
    
    # Check file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"\n✅ GIF created successfully!")
    print(f"   File size: {file_size:.2f} MB")
    if file_size > 10:
        print(f"   ⚠️  Warning: File size exceeds GitHub's 10MB limit!")
        print(f"   Consider reducing max_frames or scale further.")
    else:
        print(f"   ✅ File size is under GitHub's 10MB limit")
    
    return True

if __name__ == '__main__':
    video_file = "output_detection.mp4"
    output_file = "demo.gif"
    
    if not Path(video_file).exists():
        print(f"❌ Error: Video file '{video_file}' not found!")
        print("Please run 'python run_detection.py' first to generate the video.")
    else:
        # Create very slow GIF: 0.5 FPS (2 seconds per frame), 80 frames, 35% scale
        # This should create a ~8-9MB GIF that plays very slowly
        create_gif_from_video(video_file, output_file, max_frames=80, fps=0.5, scale=0.35)


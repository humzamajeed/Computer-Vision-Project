"""
Convert video to GIF for GitHub README display
Usage: python create_gif.py
"""

import cv2
import imageio
import numpy as np
from pathlib import Path
import sys
import io

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def create_gif_from_video(video_path, output_path, max_frames=150, fps=10, scale=0.5):
    """
    Convert video to GIF for GitHub README
    
    Args:
        video_path: Path to input video file
        output_path: Path to output GIF file
        max_frames: Maximum number of frames to include (for file size)
        fps: Frames per second for GIF
        scale: Scale factor to reduce file size (0.5 = half size)
    """
    print(f"📹 Reading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file!")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {video_fps}")
    print(f"   Total frames: {total_frames}")
    
    # Calculate frame skip to get max_frames
    frame_skip = max(1, total_frames // max_frames)
    
    print(f"\n🎬 Processing frames (every {frame_skip} frames)...")
    
    frames = []
    frame_count = 0
    saved_count = 0
    
    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames to reduce file size
        if frame_count % frame_skip == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to reduce file size
            if scale != 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            frames.append(frame_rgb)
            saved_count += 1
            
            if saved_count % 20 == 0:
                print(f"   Processed {saved_count}/{max_frames} frames...")
        
        frame_count += 1
    
    cap.release()
    
    if not frames:
        print("❌ Error: No frames extracted!")
        return
    
    print(f"\n💾 Saving GIF: {output_path}")
    print(f"   Frames: {len(frames)}")
    print(f"   FPS: {fps}")
    print(f"   This may take a minute...")
    
    # Save as GIF
    imageio.mimsave(
        output_path,
        frames,
        fps=fps,
        loop=0,  # Infinite loop
        duration=1.0/fps
    )
    
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"\n✅ GIF created successfully!")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Frames: {len(frames)}")
    print(f"\n💡 Note: GitHub has a 10MB limit for files. If GIF is too large,")
    print(f"   reduce max_frames or scale in the script.")

if __name__ == '__main__':
    video_file = "output_detection.mp4"
    output_file = "demo.gif"
    
    if not Path(video_file).exists():
        print(f"❌ Error: Video file '{video_file}' not found!")
        print("Please run 'python run_detection.py' first to generate the video.")
    else:
        # Create GIF with optimized settings for GitHub (under 10MB limit)
        # Slower playback: 70 frames, fps=4 (slower), scale=0.33 (640x360) should create ~8-9MB GIF
        create_gif_from_video(video_file, output_file, max_frames=70, fps=4, scale=0.33)


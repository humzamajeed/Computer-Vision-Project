"""
Create an animated GIF from output_detection.mp4 for README display
Includes all frames to capture license plate detections, matches video speed
Uses PIL for better compression
"""

import cv2
from PIL import Image
from pathlib import Path
import sys
import io

# Fix for UnicodeEncodeError in some environments
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def create_gif_from_video(video_path: str, output_path: str, target_width: int = 300, gif_fps: float = 5.0):
    """
    Create an animated GIF from video with all frames included, using PIL for better compression
    
    Args:
        video_path: Path to input video file
        output_path: Path to output GIF file
        target_width: Target width in pixels to match README width
        gif_fps: FPS for GIF playback (5 FPS for reasonable file size while maintaining speed)
    """
    if not Path(video_path).exists():
        print(f"❌ Error: Video file '{video_path}' not found!")
        return False
    
    print(f"🎬 Creating GIF from: {video_path}")
    print(f"   Target: {output_path}")
    print("   Including ALL frames to capture license plate detections...")
    print("   Using PIL for better compression...")
    print("   This may take a few minutes...\n")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file!")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video Info:")
    print(f"   Original FPS: {original_fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Original Resolution: {width}x{height}")
    
    # Calculate scale to match target width while maintaining aspect ratio
    scale = target_width / width
    new_width = target_width
    new_height = int(height * scale)
    
    print(f"   Scaled to: {new_width}x{new_height} (scale: {scale:.2f})")
    print(f"   GIF FPS: {gif_fps} (matches video speed perception)\n")
    
    frames = []
    frame_count = 0
    
    print("📸 Extracting ALL frames (this ensures license plate detections are included)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Include ALL frames - don't skip any
        # Resize frame to target width
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for better compression
        pil_image = Image.fromarray(rgb_frame)
        frames.append(pil_image)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"   Extracted {frame_count}/{total_frames} frames...")
    
    cap.release()
    
    if not frames:
        print("❌ Error: No frames extracted!")
        return False
    
    print(f"\n✅ Extracted all {len(frames)} frames!")
    print(f"\n💾 Saving GIF: {output_path}")
    print(f"   Frames: {len(frames)}")
    print(f"   FPS: {gif_fps} (matches video speed)")
    print(f"   Resolution: {new_width}x{new_height}")
    print("   Using PIL with optimized settings for compression...")
    print("   This may take a few minutes...")
    
    # Calculate duration per frame (in milliseconds)
    duration_ms = int(1000 / gif_fps)
    
    # Quantize frames to reduce colors (significantly reduces file size)
    print("   Quantizing colors for better compression (32 colors)...")
    quantized_frames = []
    for i, frame in enumerate(frames):
        # Convert to palette mode with very few colors for much smaller file size
        quantized = frame.quantize(method=Image.Quantize.MEDIANCUT, colors=32)
        quantized_frames.append(quantized.convert('RGB'))
        if (i + 1) % 100 == 0:
            print(f"   Quantized {i + 1}/{len(frames)} frames...")
    
    # Save GIF with PIL - better compression
    # Use optimize=True and save_all for better compression
    quantized_frames[0].save(
        output_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        method=6  # Maximum compression method
    )
    
    # Check file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"\n✅ GIF created successfully!")
    print(f"   File size: {file_size:.2f} MB")
    if file_size > 10:
        print(f"   ⚠️  Warning: File size exceeds GitHub's 10MB limit!")
        print(f"   Consider reducing target_width or gif_fps.")
        print(f"   Current: {target_width}px width, {gif_fps} FPS")
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
        # Create GIF with ALL frames, matching video speed
        # Using 180px width and 3 FPS with very aggressive color quantization (32 colors) to keep under 10MB
        # 3 FPS gives reasonable playback speed, 32 colors significantly reduces file size
        create_gif_from_video(video_file, output_file, target_width=180, gif_fps=3.0)

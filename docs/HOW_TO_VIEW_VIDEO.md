# 🎬 How to View and Verify Your Saved Video

## Quick Methods to View the Video

### Method 1: Use the Verification Script (Recommended)

I've created a simple Python script to verify and view your video:

```bash
python verify_video.py
```

**Features:**
- ✅ Shows video properties (resolution, FPS, duration, file size)
- ✅ Plays the video with controls
- ✅ Press `q` to quit
- ✅ Press `SPACE` to pause/resume
- ✅ Press `f` to toggle fullscreen

### Method 2: Use Any Video Player

Simply double-click `output_detection.mp4` or open it with:

**Windows:**
- Windows Media Player (built-in)
- VLC Media Player (recommended - free)
- Windows Movies & TV

**To open with VLC:**
1. Download VLC from https://www.videolan.org/
2. Right-click `output_detection.mp4`
3. Select "Open with" → "VLC media player"

**Mac:**
- QuickTime Player (built-in)
- VLC Media Player

**Linux:**
- VLC Media Player
- MPV
- Totem

### Method 3: Check Video Properties (Without Playing)

**Windows (PowerShell):**
```powershell
Get-Item output_detection.mp4 | Select-Object Name, Length, LastWriteTime
```

**Python Script:**
```python
from pathlib import Path
import cv2

video_file = "output_detection.mp4"
cap = cv2.VideoCapture(video_file)

if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Frames: {frames}")
    print(f"Duration: {frames/fps:.2f} seconds")
    print(f"File size: {Path(video_file).stat().st_size / (1024*1024):.2f} MB")
    
    cap.release()
else:
    print("Error: Could not open video")
```

## What to Look For in the Video

When viewing the video, you should see:

✅ **Green boxes** around vehicles with labels like:
   - `car: 0.89`
   - `truck: 0.95`
   - `bus: 0.87`

✅ **Blue boxes** around license plates with labels like:
   - `Plate: 0.85`
   - `Plate: 0.92`

✅ **Info overlay** in the top-left corner:
   - `Frame: 150/500 | Vehicles: 2 | Plates: 1`

✅ **Legend** at the bottom:
   - `Green = Vehicle | Blue = License Plate`

## Troubleshooting

### Video won't play?

1. **Check if file exists:**
   ```bash
   dir output_detection.mp4  # Windows
   ls -lh output_detection.mp4  # Linux/Mac
   ```

2. **Check file size:**
   - If file is 0 bytes, the video wasn't saved correctly
   - Re-run `python run_detection.py`

3. **Try different player:**
   - Use VLC (works with most codecs)
   - Or use the verification script

### Video is corrupted?

- Make sure you didn't interrupt the script (press 'q' to quit properly)
- Re-run the detection script
- Check disk space

### Video has no detections?

- Check that the input video has vehicles
- Lower the confidence threshold in `run_detection.py`:
  ```python
  vehicles = vehicle_detector.detect_vehicles(frame, conf_threshold=0.15, imgsz=416)
  ```

## Quick Verification Checklist

- [ ] File `output_detection.mp4` exists
- [ ] File size > 0 bytes
- [ ] Video plays in a media player
- [ ] Green boxes visible on vehicles
- [ ] Blue boxes visible on license plates (if plates are detected)
- [ ] Video has same duration as input video
- [ ] Video resolution matches input video

## Command Line Quick Check

**Windows:**
```cmd
python verify_video.py
```

**Or just check if file exists:**
```cmd
dir output_detection.mp4
```

**Linux/Mac:**
```bash
python verify_video.py
```

**Or:**
```bash
ls -lh output_detection.mp4
ffprobe output_detection.mp4  # If ffmpeg is installed
```

---

**Tip:** The verification script (`verify_video.py`) is the easiest way to check everything at once!


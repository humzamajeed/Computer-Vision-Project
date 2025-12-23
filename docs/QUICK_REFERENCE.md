# 🚀 Quick Reference Guide

## 📋 Project Summary

**What it does**: Detects vehicles and license plates in video footage  
**Main file**: `run_detection.py`  
**Technologies**: YOLO11 (vehicles) + OpenCV (plates)

---

## 🔑 Key Files

| File | Purpose |
|------|---------|
| `run_detection.py` | Main script - runs detection on video |
| `vehicle_detector.py` | Vehicle detection using YOLO11 |
| `license_plate_detector.py` | License plate detection using CV methods |
| `model_comparison.py` | Compares different YOLO models |

---

## 🎯 How It Works (Simple)

```
1. Load video
2. For each frame:
   ├─→ Detect vehicles (YOLO11) → Green boxes
   └─→ For each vehicle:
       └─→ Detect license plate (CV methods) → Blue box
3. Display results
```

---

## 🔧 Main Functions

### Vehicle Detection
```python
vehicle_detector = create_detector('yolo11', 'n')
vehicles = vehicle_detector.detect_vehicles(frame, conf_threshold=0.25)
# Returns: [{'bbox': [x1,y1,x2,y2], 'confidence': 0.89, 'class_name': 'car'}, ...]
```

### License Plate Detection
```python
plate_detector = LicensePlateDetector(debug=True)
plate_info = plate_detector.process_vehicle(frame, vehicle_bbox)
# Returns: {'plate_bbox': [x1,y1,x2,y2] or None, 'confidence': 0.85}
```

---

## 📊 Detection Methods

### Vehicle Detection
- **Method**: YOLO11 Neural Network
- **Classes**: Car (2), Motorcycle (3), Bus (5), Truck (7)
- **Output**: Bounding box + confidence score

### License Plate Detection
- **Method 1**: MSER (text region detection)
- **Method 2**: Adaptive Thresholding (3 variants)
- **Method 3**: Canny Edge Detection
- **Validation**: Aspect ratio, size, contrast, text structure, position

---

## ⚙️ Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `conf_threshold` | 0.25 | Minimum 25% confidence |
| `imgsz` | 416 | Input size (416 = faster) |
| `min_aspect_ratio` | 2.5 | Plates are 2.5x wider than tall |
| `max_aspect_ratio` | 4.2 | Plates are max 4.2x wider than tall |
| `min_score` | 0.25 | Minimum detection score |
| `min_width` | 80 | Minimum plate width (pixels) |
| `min_height` | 25 | Minimum plate height (pixels) |

---

## 🎨 Visual Output

- **Green Box**: Vehicle (`car: 0.89`)
- **Blue Box**: License Plate (`Plate: 0.85`)
- **Top-left**: Frame counter and statistics
- **Bottom**: Legend

---

## 🚀 Running

```bash
# Install
pip install -r requirements.txt

# Run
python run_detection.py

# Compare models (optional)
python model_comparison.py
```

**Controls**:
- `q` = Quit
- `p` = Pause/Resume

---

## 🔍 Detection Flow

```
Frame
  ↓
[YOLO11] → Vehicles (green boxes)
  ↓
For each vehicle:
  Crop ROI
    ↓
  [MSER + Adaptive + Canny] → Plate candidates
    ↓
  [Validate] → Filter false positives
    ↓
  [Score] → Best candidate
    ↓
  If score > 0.25 → Plate (blue box)
```

---

## 📁 Project Structure

```
parking-management-system-using-CV/
├── run_detection.py              # Main script
├── vehicle_detection/
│   ├── models/
│   │   └── vehicle_detector.py  # YOLO11 vehicle detection
│   └── license_plate/
│       └── license_plate_detector.py  # CV plate detection
└── requirements.txt
```

---

## 🐛 Debug Mode

When `debug=True`:
- Saves detected plates to `debug_plates/` folder
- Files: `plate_0_detected.jpg`, `plate_1_detected.jpg`, etc.

---

## 💡 Key Concepts

1. **Two-Stage Detection**: Vehicles first, then plates within vehicles
2. **Multiple Methods**: 3+ CV methods for robustness
3. **Strict Validation**: Filters false positives (windows)
4. **Region Focusing**: Only searches lower 40% (where plates are)
5. **Score Threshold**: Only high-confidence detections (0.25)

---

## 📝 Common Issues

| Issue | Solution |
|-------|----------|
| No vehicles detected | Lower `conf_threshold` (e.g., 0.15) |
| No plates detected | Check `debug_plates/` folder, adjust thresholds |
| Too slow | Reduce `imgsz` to 320 or 256 |
| False positives | Increase `min_score` threshold |

---

## 🔗 Dependencies

- `ultralytics` - YOLO11 models
- `opencv-python` - Video I/O and CV operations
- `numpy` - Array operations
- `torch` - Deep learning framework

---

For detailed explanation, see `PROJECT_EXPLANATION.md`




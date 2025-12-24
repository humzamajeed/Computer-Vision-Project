# 🚗 Intelligent Parking Management System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-11-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A Computer Vision-powered system for automated vehicle and license plate detection in parking environments**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Demo](#-demo-video) • [Project Structure](#-project-structure)

</div>

---

## 📝 Summary

This project implements an intelligent parking management system that leverages state-of-the-art deep learning and computer vision techniques to automatically detect vehicles and their license plates in video footage. The system uses YOLO11 neural networks for robust vehicle detection and advanced CV algorithms for precise license plate localization, making it ideal for automated parking monitoring, security systems, and traffic analysis applications.

---

## 🎥 Demo Video

<div align="center">

**Watch the system in action: Real-time vehicle and license plate detection on parking garage footage**

![Demo GIF](demo.gif)

*Animated GIF showing vehicle and license plate detection results (464px width, optimized for README display)*

</div>

### 📹 Full Output Video

<div align="center">

**[🎬 Watch on YouTube](https://youtube.com/shorts/lVSzGinErlw)** | **[📥 Download Video](output_detection.mp4)**

*Click the YouTube link to watch the full processed video, or download the MP4 file directly*

</div>

**Demo Description:**

This video demonstrates the system processing a parking garage security camera feed. The detection pipeline works in two stages:

1. **Vehicle Detection** (🟢 Green boxes): The YOLO11 neural network identifies vehicles in real-time, displaying the vehicle type (car, truck, bus, motorcycle) and detection confidence score.

2. **License Plate Detection** (🔵 Blue boxes): For each detected vehicle, advanced computer vision algorithms locate the license plate using multiple detection methods (MSER, adaptive thresholding, Canny edge detection) with strict validation to avoid false positives.

The overlay shows real-time statistics including frame count and detection metrics, demonstrating the system's ability to process video streams efficiently while maintaining high accuracy.

---

## ✨ Features

- **🚗 Vehicle Detection**: State-of-the-art YOLO11 neural network for detecting cars, trucks, buses, and motorcycles
- **🔍 License Plate Detection**: Multi-method computer vision approach using MSER, adaptive thresholding, and Canny edge detection
- **🎯 High Accuracy**: Advanced validation filters to minimize false positives (e.g., distinguishing plates from car windows)
- **⚡ Real-time Processing**: Optimized for real-time video analysis with configurable performance settings
- **📹 Video I/O**: Automatic video processing with output generation for analysis and demonstration
- **🔧 Modular Architecture**: Clean, extensible codebase with separate modules for vehicle and plate detection

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+** - Programming language
- **OpenCV (cv2)** - Computer vision operations, video I/O, and image processing
- **NumPy** - Numerical computations and array operations

### Deep Learning
- **Ultralytics YOLO11** - State-of-the-art object detection model for vehicle recognition
- **PyTorch** - Deep learning framework for neural network inference
- **TorchVision** - Computer vision utilities

### Computer Vision Techniques
- **MSER (Maximally Stable Extremal Regions)** - Text region detection
- **Adaptive Thresholding** - Multi-variant thresholding (Gaussian, Mean, Otsu)
- **Canny Edge Detection** - Edge-based feature extraction
- **Morphological Operations** - Image processing for text region enhancement

### Additional Tools
- **Pandas** - Data analysis and model comparison results
- **Pathlib** - Modern file path handling

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/parking-management-system-using-CV.git
cd parking-management-system-using-CV
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: The YOLO models will be automatically downloaded on first run (approximately 6MB for YOLO11 Nano).

---

## 🚀 Usage

### Basic Usage

1. **Prepare your video file** (optional - default video will be used if available):
   - Place your video file in the project root directory
   - Update the `video_file` variable in `run_detection.py` if using a custom video

2. **Run the detection script**:
   ```bash
   python run_detection.py
   ```

3. **Controls during execution**:
   - Press `q` to quit
   - Press `p` to pause/resume processing

4. **Output**:
   - Real-time display window with detection results

### Advanced Usage

**Model Comparison Tool**:
```bash
python model_comparison.py --video your_video.mp4
```

### Configuration

You can modify detection parameters in `run_detection.py`:

```python
# Vehicle detection confidence threshold (0.0 - 1.0)
vehicles = vehicle_detector.detect_vehicles(frame, conf_threshold=0.25, imgsz=416)

# License plate detection confidence threshold
plate_info = plate_detector.process_vehicle(frame, vehicle['bbox'], conf_threshold=0.25)
```

---

## 📁 Project Structure

```
parking-management-system-using-CV/
│
├── 📄 README.md                          # Project documentation
├── 📄 requirements.txt                   # Python dependencies
│
├── 🎬 demo.gif                          # Animated demo GIF (for README)
├── 🎥 output_detection.mp4              # Full processed output video
│
├── 🐍 run_detection.py                   # Main execution script
├── 🐍 model_comparison.py                # Model evaluation tool
├── 🐍 create_gif.py                      # Utility to create demo GIF from output video
│
└── 📦 vehicle_detection/                 # Core detection modules
    ├── __init__.py
    │
    ├── 📁 models/                        # Vehicle detection module
    │   ├── __init__.py
    │   └── vehicle_detector.py           # YOLO11 vehicle detector
    │
    └── 📁 license_plate/                 # License plate detection module
        ├── __init__.py
        └── license_plate_detector.py     # CV-based plate detector
```


### Key Files Explained

| File | Description |
|------|-------------|
| `run_detection.py` | Main script that orchestrates video processing, vehicle detection, and license plate detection. Saves output to `output_detection.mp4` |
| `vehicle_detector.py` | Implements YOLO11-based vehicle detection with configurable model sizes |
| `license_plate_detector.py` | Multi-method CV approach for license plate detection with strict validation |
| `model_comparison.py` | Tool for comparing different YOLO model variants (YOLOv5, YOLOv8, YOLO11) |
| `create_gif.py` | Utility script to create slow-motion demo GIF from `output_detection.mp4` for README display |

---

## 🔬 How It Works

### Detection Pipeline

```
Input Video Frame
    ↓
[Stage 1: Vehicle Detection]
    ├─→ YOLO11 Neural Network Inference
    ├─→ Filter by Vehicle Classes (car, truck, bus, motorcycle)
    └─→ Output: Bounding boxes with confidence scores
    ↓
[Stage 2: License Plate Detection]
    For each detected vehicle:
    ├─→ Crop vehicle region (ROI)
    ├─→ Apply multiple CV methods:
    │   ├─→ MSER (text region detection)
    │   ├─→ Adaptive Thresholding (3 variants)
    │   └─→ Canny Edge Detection
    ├─→ Validate candidates:
    │   ├─→ Aspect ratio (2.5:1 to 4.2:1)
    │   ├─→ Size constraints (80x25 min, 50% max width)
    │   ├─→ Contrast analysis (>25)
    │   ├─→ Text structure validation (horizontal bands)
    │   └─→ Position validation (centered, lower region)
    └─→ Score and return best candidate
    ↓
Output: Annotated frame with vehicle and plate bounding boxes
```

### Key Algorithms

1. **Vehicle Detection**: YOLO11 (You Only Look Once) - Single-stage object detector optimized for speed and accuracy
2. **License Plate Detection**: Hybrid approach combining:
   - **MSER**: Detects stable regions characteristic of text
   - **Adaptive Thresholding**: Handles varying lighting conditions
   - **Canny Edge Detection**: Identifies edge patterns in plate regions
3. **Validation Pipeline**: Multi-stage filtering to reduce false positives by analyzing aspect ratio, size, contrast, text structure, and spatial positioning

---

## 📊 Performance

- **Vehicle Detection**: ~30-60 FPS (depending on hardware and input resolution)
- **License Plate Detection**: Optimized for accuracy with configurable speed/accuracy trade-offs
- **Model Size**: YOLO11 Nano (~6MB) - lightweight and efficient
- **Accuracy**: High precision with strict validation filters to minimize false positives

---

## 🎓 Academic Context

This project was developed as part of a **5th-semester Computer Vision course**, demonstrating practical application of:
- Deep learning for object detection
- Traditional computer vision techniques
- Video processing and analysis
- Software engineering best practices

---

---

## 👤 Author

**Muhammad Humza Majeed**

---

## 🙏 Acknowledgments

- **Ultralytics** for the YOLO11 implementation
- **OpenCV** community for excellent computer vision tools
- **PyTorch** team for the deep learning framework

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

Made using Python and Computer Vision

</div>

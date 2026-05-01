# 🌴 Coconut Tree Detection and Counting

A comprehensive computer vision project for detecting and counting coconut trees in aerial/satellite images using advanced deep learning techniques.

## 📊 Results
- **Detection Accuracy**: 581 coconut trees detected in test image
- **Methods**: YOLO + Contours + Watershed + Edges + Green Channel Analysis
- **Image Size**: 954x711 pixels

## 📁 Project Structure

```
coconut_tree_counter/
├── src/
│   ├── main.py              # Main detection script
│   └── utils/               # Detection utilities
│       ├── tiling.py        # Image tiling for large images
│       ├── nms.py           # Non-Maximum Suppression
│       ├── draw_boxes.py    # Visualization
│       ├── ultra_aggressive.py  # Advanced detection methods
│       ├── advanced_detection.py
│       ├── refined_detection.py
│       └── __init__.py
├── data/
│   ├── input/               # Input images
│   │   └── test2.png        # Test image (954x711)
│   └── models/              # Pre-trained models
│       └── palm.pt          # YOLOv8 coconut palm model
├── results/
│   └── detected/            # Detection results
│       ├── detected.png     # Annotated image with detections
│       └── count.txt        # Detection statistics
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Detection
```bash
python src/main.py
```

### 3. View Results
- **Image**: `results/detected/detected.png`
- **Count**: `results/detected/count.txt`

## 📦 Requirements
- Python 3.8+
- ultralytics (YOLOv8)
- OpenCV
- NumPy
- Torch
- TorchVision

See `requirements.txt` for complete list.

## 🔬 Detection Methods

The system uses a hybrid approach:
1. **YOLO Model** - Deep learning object detection
2. **Contour Detection** - Edge-based tree identification
3. **Watershed Segmentation** - Separating overlapping trees
4. **Edge Detection** - Boundary analysis
5. **Green Channel Analysis** - Vegetation detection

## 📈 Performance
- **Input**: 954x711 pixel aerial image
- **Processing**: Direct (no tiling required)
- **Output**: 581 coconut trees detected
- **Confidence**: Multi-method validation

## 📝 Notes
- Model trained specifically for coconut palm detection
- Results optimized for tropical plantation imagery
- Supports both small and large images with automatic tiling

## 👤 Author
Roopesh

## 📅 Updated
May 2026

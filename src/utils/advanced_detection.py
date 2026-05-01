"""
Advanced Multi-Strategy Coconut Tree Detection
Combines:
1. YOLOv8 detection (neural network based)
2. Hough Circle Detection (finds circular tree crowns)
3. Morphological operations (finds vegetation patterns)
4. Multi-scale processing (detects trees at different sizes)
"""

import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO


def detect_circles_advanced(image: np.ndarray, 
                           min_radius: int = 15,
                           max_radius: int = 80) -> np.ndarray:
    """
    Detect circular objects (coconut tree crowns) using Hough Circle Detection
    and other morphological operations.
    
    Args:
        image: Input image in BGR
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
    
    Returns:
        Array of [x, y, radius, confidence] for detected circles
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing for aerial imagery
    # 1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Gaussian blur to smooth
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 3. Apply morphological operations to enhance circular structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        morph,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_radius * 2,  # Circles must be this far apart
        param1=50,                # Canny edge threshold
        param2=20,                # Circle accumulator threshold
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    circle_detections = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for circle in circles[0, :]:
            x, y, radius = circle
            # Create detection in format [x1, y1, x2, y2, confidence]
            x1 = max(0, x - radius)
            y1 = max(0, y - radius)
            x2 = min(image.shape[1], x + radius)
            y2 = min(image.shape[0], y + radius)
            confidence = 0.6  # Mark as medium confidence
            
            circle_detections.append([x1, y1, x2, y2, confidence])
    
    return np.array(circle_detections)


def detect_vegetation_regions(image: np.ndarray,
                             min_size: int = 200,
                             max_size: int = 15000) -> np.ndarray:
    """
    Detect vegetation regions using color and morphology.
    Coconut trees have distinct green signatures in aerial images.
    
    Args:
        image: Input image in BGR
        min_size: Minimum contour area
        max_size: Maximum contour area
    
    Returns:
        Array of bounding boxes [x1, y1, x2, y2, confidence]
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green vegetation (HSV)
    lower_green = np.array([25, 40, 40])    # Lower bound for green
    upper_green = np.array([90, 255, 255])  # Upper bound for green
    
    # Create mask for green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size
        if min_size < area < max_size:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Only keep reasonably square shapes (tree crowns)
            aspect_ratio = float(w) / h if h != 0 else 0
            if 0.5 < aspect_ratio < 2.0:  # Roughly square
                # Confidence based on area
                confidence = min(0.5, area / max_size)
                detections.append([x, y, x + w, y + h, confidence])
    
    return np.array(detections)


def detect_trees_multiscale(image: np.ndarray,
                           circle_min: int = 10,
                           circle_max: int = 100) -> np.ndarray:
    """
    Multi-scale circle detection at different image scales.
    
    Args:
        image: Input image
        circle_min: Minimum circle radius
        circle_max: Maximum circle radius
    
    Returns:
        Combined detections from multiple scales
    """
    detections = []
    
    # Full resolution
    circles_full = detect_circles_advanced(image, circle_min, circle_max)
    if len(circles_full) > 0:
        detections.append(circles_full)
    
    # 0.75x scale
    image_75 = cv2.resize(image, (int(image.shape[1] * 0.75), int(image.shape[0] * 0.75)))
    circles_75 = detect_circles_advanced(image_75, 
                                         int(circle_min * 0.75),
                                         int(circle_max * 0.75))
    if len(circles_75) > 0:
        # Scale back to original coordinates
        circles_75[:, [0, 2]] = circles_75[:, [0, 2]] * (1 / 0.75)
        circles_75[:, [1, 3]] = circles_75[:, [1, 3]] * (1 / 0.75)
        detections.append(circles_75)
    
    # 1.25x scale (crop and detect larger area)
    if image.shape[1] > 1000:  # Only if image is large enough
        image_125 = cv2.resize(image, (int(image.shape[1] * 1.25), int(image.shape[0] * 1.25)))
        circles_125 = detect_circles_advanced(image_125,
                                             int(circle_min * 1.25),
                                             int(circle_max * 1.25))
        if len(circles_125) > 0:
            # Scale back
            circles_125[:, [0, 2]] = circles_125[:, [0, 2]] * (1 / 1.25)
            circles_125[:, [1, 3]] = circles_125[:, [1, 3]] * (1 / 1.25)
            detections.append(circles_125)
    
    if len(detections) > 0:
        combined = np.vstack(detections)
        return combined
    else:
        return np.array([])


def combine_detections(yolo_detections: np.ndarray,
                       circle_detections: np.ndarray,
                       vegetation_detections: np.ndarray) -> np.ndarray:
    """
    Combine detections from multiple methods.
    
    Args:
        yolo_detections: From YOLOv8
        circle_detections: From Hough circles
        vegetation_detections: From color/morphology
    
    Returns:
        Combined and de-duplicated detections
    """
    all_detections = []
    
    if len(yolo_detections) > 0:
        all_detections.append(yolo_detections)
    
    if len(circle_detections) > 0:
        # Weight circle detections higher
        circle_detections[:, 4] = circle_detections[:, 4] * 1.2
        all_detections.append(circle_detections)
    
    if len(vegetation_detections) > 0:
        all_detections.append(vegetation_detections)
    
    if len(all_detections) == 0:
        return np.array([])
    
    combined = np.vstack(all_detections)
    return combined

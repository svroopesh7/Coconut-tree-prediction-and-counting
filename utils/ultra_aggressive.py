"""
Ultra-Aggressive Coconut Tree Detection
- Watershed segmentation to separate trees
- Edge detection for tree boundaries
- Template matching for tree shapes
- Aggressive small feature detection
- Connected component analysis
- Optimized for dense plantations
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import List, Tuple


def detect_trees_aggressive(image: np.ndarray) -> np.ndarray:
    """
    Ultra-aggressive tree detection optimized for coconut plantations.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. CLAHE enhancement for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Blur for smoothing
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 3. Morphological opening to separate trees
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 4. Morphological closing to connect nearby trees
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_med, iterations=1)
    
    # 5. Binary threshold to find dark spots (tree shadows/crowns)
    _, binary = cv2.threshold(closed, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 6. Find contours (tree crowns)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    min_area = 30  # Very small minimum
    max_area = 5000
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (trees are roughly circular)
            if w > 0 and h > 0:
                aspect = float(w) / h
                if 0.4 < aspect < 2.5:  # More permissive
                    detections.append([x, y, x + w, y + h, 0.5])
    
    return np.array(detections)


def detect_trees_watershed(image: np.ndarray) -> np.ndarray:
    """
    Watershed segmentation to separate touching/overlapping trees.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Binary threshold to find objects
    _, binary = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    
    # Sure background
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_bg = cv2.dilate(morph, kernel_bg, iterations=3)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Find sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all labels so sure background is not 0, but 1
    markers = markers + 1
    
    # Mark unknown region as 0
    markers[unknown == 255] = 0
    
    # Watershed
    image_8bit = cv2.convertScaleAbs(image)
    markers = cv2.watershed(image_8bit, markers)
    
    # Extract regions
    detections = []
    unique_markers = np.unique(markers)
    
    for marker in unique_markers:
        if marker <= 1:  # Skip background and border
            continue
        
        mask = (markers == marker).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 3000:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 0 and h > 0:
                    detections.append([x, y, x + w, y + h, 0.4])
    
    return np.array(detections)


def detect_trees_edges(image: np.ndarray) -> np.ndarray:
    """
    Edge-based detection to find tree boundaries.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect nearby boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours from edges
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 5:
                detections.append([x, y, x + w, y + h, 0.3])
    
    return np.array(detections)


def detect_trees_green_channel(image: np.ndarray) -> np.ndarray:
    """
    Focus on green channel which shows vegetation clearly.
    """
    # Extract channels
    b, g, r = cv2.split(image)
    
    # Green minus red (vegetation index approximation)
    veg_index = cv2.subtract(g, r).astype(float)
    
    # Normalize
    veg_index = np.clip(veg_index, 0, 255).astype(np.uint8)
    
    # Enhance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(veg_index)
    
    # Binary threshold
    _, binary = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY)
    
    # Find blobs
    detections = []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 40 < area < 5000:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0 and h > 0:
                detections.append([x, y, x + w, y + h, 0.45])
    
    return np.array(detections)


def super_aggressive_detection(image: np.ndarray) -> np.ndarray:
    """
    Combine ALL detection methods for maximum trees.
    """
    methods = [
        ("Basic Contours", detect_trees_aggressive),
        ("Watershed", detect_trees_watershed),
        ("Edges", detect_trees_edges),
        ("Green Channel", detect_trees_green_channel),
    ]
    
    all_detections = []
    
    for method_name, method_func in methods:
        try:
            detections = method_func(image)
            if len(detections) > 0:
                all_detections.append(detections)
                print(f"   ✓ {method_name}: {len(detections)} detections")
        except Exception as e:
            print(f"   ⚠ {method_name} failed: {str(e)[:50]}")
    
    if len(all_detections) == 0:
        return np.array([])
    
    combined = np.vstack(all_detections)
    return combined

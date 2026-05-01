"""
REFINED Coconut Tree Detection - INDIVIDUAL TREES ONLY
Focuses on detecting SINGLE trees, not plantation blocks
Key: Filter by MAX BOX SIZE to avoid grouping multiple trees
"""

import cv2
import numpy as np
from scipy import ndimage


def detect_individual_trees(image: np.ndarray, 
                            min_size: int = 30,
                            max_size: int = 250) -> np.ndarray:
    """
    Detect INDIVIDUAL tree crowns - not plantation blocks.
    
    Args:
        image: Input image
        min_size: Minimum tree size (pixels)
        max_size: MAXIMUM tree size - NO HUGE BOXES! (pixels)
    
    Returns:
        Detection boxes [x1, y1, x2, y2, confidence]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Binary threshold
    _, binary = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Small morphology to separate trees
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # STRICT SIZE FILTERING - only individual trees
        if min_size < area < max_size:
            x, y, w, h = cv2.boundingRect(contour)
            
            # No huge boxes!
            if w > 0 and h > 0 and w < 200 and h < 200:
                # Filter by aspect ratio (roughly circular trees)
                aspect = float(w) / h if h > 0 else 0
                if 0.5 < aspect < 2.0:
                    confidence = 0.5
                    detections.append([x, y, x + w, y + h, confidence])
    
    return np.array(detections)


def detect_trees_green_strict(image: np.ndarray,
                              min_size: int = 40,
                              max_size: int = 250) -> np.ndarray:
    """
    Detect individual trees using GREEN BAND extraction.
    Much more selective.
    """
    b, g, r = cv2.split(image)
    
    # Focus on green channel (vegetation)
    green_enhanced = g.astype(float) - 0.5 * (r.astype(float) + b.astype(float))
    green_enhanced = np.clip(green_enhanced, 0, 255).astype(np.uint8)
    
    # Threshold
    _, binary = cv2.threshold(green_enhanced, 50, 255, cv2.THRESH_BINARY)
    
    # Small morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_size < area < max_size:
            x, y, w, h = cv2.boundingRect(contour)
            
            if 0 < w < 200 and 0 < h < 200:
                aspect = float(w) / h if h > 0 else 0
                if 0.4 < aspect < 2.5:
                    detections.append([x, y, x + w, y + h, 0.45])
    
    return np.array(detections)


def detect_trees_blob_analysis(image: np.ndarray,
                               min_size: int = 50,
                               max_size: int = 280) -> np.ndarray:
    """
    Blob detection to find circular objects (trees).
    Only small blobs = individual trees.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Try to find dark circular regions (tree shadows)
    _, binary = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Label connected components
    labeled, num_features = ndimage.label(binary)
    
    detections = []
    
    for label_id in range(1, num_features + 1):
        component = (labeled == label_id).astype(np.uint8) * 255
        area = np.sum(component > 0)
        
        if min_size < area < max_size:
            y_indices, x_indices = np.where(component > 0)
            
            if len(x_indices) > 0:
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                w = x_max - x_min
                h = y_max - y_min
                
                if 0 < w < 200 and 0 < h < 200:
                    detections.append([x_min, y_min, x_max, y_max, 0.4])
    
    return np.array(detections)


def detect_trees_edges_strict(image: np.ndarray,
                              min_size: int = 80,
                              max_size: int = 300) -> np.ndarray:
    """
    Edge-based detection, strict filtering for individual trees.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Canny edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Small dilation to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_size < area < max_size:
            x, y, w, h = cv2.boundingRect(contour)
            if 0 < w < 200 and 0 < h < 200:
                detections.append([x, y, x + w, y + h, 0.35])
    
    return np.array(detections)


def refined_detection_all_methods(image: np.ndarray) -> np.ndarray:
    """
    Combine all refined detection methods.
    STRICT SIZE FILTERING - only individual trees, no huge boxes.
    """
    methods = [
        ("Individual Contours", lambda img: detect_individual_trees(img, 30, 250)),
        ("Green Channel Strict", lambda img: detect_trees_green_strict(img, 40, 250)),
        ("Blob Analysis", lambda img: detect_trees_blob_analysis(img, 50, 280)),
        ("Edge Detection Strict", lambda img: detect_trees_edges_strict(img, 80, 300)),
    ]
    
    all_detections = []
    
    for method_name, method_func in methods:
        try:
            detections = method_func(image)
            if len(detections) > 0:
                all_detections.append(detections)
                print(f"   ✓ {method_name}: {len(detections)} INDIVIDUAL trees")
        except Exception as e:
            print(f"   ⚠ {method_name} failed")
    
    if len(all_detections) == 0:
        return np.array([])
    
    combined = np.vstack(all_detections)
    return combined

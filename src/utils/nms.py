"""
Non-Maximum Suppression (NMS) module for removing duplicate detections.
Handles merging of detections from overlapping tiles.
"""

import numpy as np
from typing import List


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (np.ndarray): First box [x1, y1, x2, y2, confidence]
        box2 (np.ndarray): Second box [x1, y1, x2, y2, confidence]
    
    Returns:
        float: IoU value between 0 and 1
    
    Example:
        iou = calculate_iou(box1, box2)
        if iou > 0.5:
            # Boxes overlap significantly
    """
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def apply_nms(boxes: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Apply Non-Maximum Suppression to remove duplicate bounding boxes.
    Keeps boxes with highest confidence scores.
    
    Args:
        boxes (np.ndarray): Bounding boxes (N x 5) - [x1, y1, x2, y2, confidence]
        iou_threshold (float): IoU threshold for suppression (default: 0.5)
    
    Returns:
        np.ndarray: Filtered bounding boxes after NMS
    
    Example:
        filtered_boxes = apply_nms(all_boxes, iou_threshold=0.5)
    """
    if len(boxes) == 0:
        return boxes
    
    # Sort boxes by confidence score in descending order
    sorted_indices = np.argsort(-boxes[:, 4])
    sorted_boxes = boxes[sorted_indices]
    
    keep_indices = []
    
    while len(sorted_boxes) > 0:
        # Keep the box with highest confidence
        keep_indices.append(sorted_indices[0])
        
        if len(sorted_boxes) == 1:
            break
        
        # Get the kept box
        kept_box = sorted_boxes[0]
        
        # Calculate IoU with all other boxes
        ious = np.array([calculate_iou(kept_box, box) for box in sorted_boxes[1:]])
        
        # Keep only boxes with IoU below threshold
        mask = ious < iou_threshold
        sorted_boxes = sorted_boxes[1:][mask]
        sorted_indices = sorted_indices[1:][np.where(mask)[0]]
    
    # Return boxes in original order
    return boxes[np.array(keep_indices)]


def merge_detections(all_detections: List[np.ndarray]) -> np.ndarray:
    """
    Merge detections from multiple tiles/images into a single array.
    
    Args:
        all_detections (List[np.ndarray]): List of detection arrays
    
    Returns:
        np.ndarray: Merged detections array (N x 5)
    
    Example:
        all_boxes = [tile1_boxes, tile2_boxes, tile3_boxes]
        merged = merge_detections(all_boxes)
    """
    if len(all_detections) == 0:
        return np.array([])
    
    # Filter out empty detections
    valid_detections = [det for det in all_detections if len(det) > 0]
    
    if len(valid_detections) == 0:
        return np.array([])
    
    # Concatenate all detections
    merged = np.vstack(valid_detections)
    return merged

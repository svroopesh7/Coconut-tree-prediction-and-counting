"""
Tiling module for splitting large images into smaller tiles.
Handles both tile creation and coordinate conversion from tile to image space.
"""

import numpy as np
from typing import List, Tuple


def tile_image(image: np.ndarray, tile_size: int = 640) -> List[Tuple[np.ndarray, int, int]]:
    """
    Split a large image into overlapping tiles for processing.
    
    Args:
        image (np.ndarray): Input image (H x W x C)
        tile_size (int): Size of each tile (default: 640x640)
    
    Returns:
        List[Tuple[np.ndarray, int, int]]: List of (tile, start_x, start_y) tuples
    
    Example:
        tiles = tile_image(image, tile_size=640)
        for tile, x, y in tiles:
            # Process tile
    """
    height, width = image.shape[:2]
    tiles = []
    
    # Calculate overlap to ensure no missed detections at tile boundaries
    overlap = 50  # pixels to overlap between tiles
    stride = tile_size - overlap
    
    # Create tiles with overlap
    y_positions = list(range(0, height - tile_size, stride)) + [max(0, height - tile_size)]
    x_positions = list(range(0, width - tile_size, stride)) + [max(0, width - tile_size)]
    
    # Remove duplicates while preserving order
    y_positions = list(dict.fromkeys(y_positions))
    x_positions = list(dict.fromkeys(x_positions))
    
    for y_start in y_positions:
        for x_start in x_positions:
            # Extract tile
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)
            
            tile = image[y_start:y_end, x_start:x_end]
            tiles.append((tile, x_start, y_start))
    
    return tiles


def convert_tile_boxes_to_image_coords(
    boxes: np.ndarray, 
    tile_x: int, 
    tile_y: int
) -> np.ndarray:
    """
    Convert bounding box coordinates from tile space to original image space.
    
    Args:
        boxes (np.ndarray): Bounding boxes in tile coordinates (N x 5)
                           Format: [x1, y1, x2, y2, confidence]
        tile_x (int): X offset of tile in original image
        tile_y (int): Y offset of tile in original image
    
    Returns:
        np.ndarray: Bounding boxes in image coordinates (N x 5)
    
    Example:
        image_boxes = convert_tile_boxes_to_image_coords(tile_boxes, 100, 100)
    """
    if len(boxes) == 0:
        return boxes
    
    converted_boxes = boxes.copy()
    converted_boxes[:, 0] += tile_x  # x1
    converted_boxes[:, 1] += tile_y  # y1
    converted_boxes[:, 2] += tile_x  # x2
    converted_boxes[:, 3] += tile_y  # y2
    
    return converted_boxes


def clip_boxes_to_image(boxes: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
    """
    Clip bounding boxes to image boundaries to ensure they stay within image dimensions.
    
    Args:
        boxes (np.ndarray): Bounding boxes (N x 5) - [x1, y1, x2, y2, confidence]
        image_height (int): Height of original image
        image_width (int): Width of original image
    
    Returns:
        np.ndarray: Clipped bounding boxes
    """
    if len(boxes) == 0:
        return boxes
    
    clipped_boxes = boxes.copy()
    clipped_boxes[:, 0] = np.clip(clipped_boxes[:, 0], 0, image_width)   # x1
    clipped_boxes[:, 1] = np.clip(clipped_boxes[:, 1], 0, image_height)  # y1
    clipped_boxes[:, 2] = np.clip(clipped_boxes[:, 2], 0, image_width)   # x2
    clipped_boxes[:, 3] = np.clip(clipped_boxes[:, 3], 0, image_height)  # y2
    
    return clipped_boxes

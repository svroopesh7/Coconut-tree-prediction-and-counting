"""
Drawing module for visualizing bounding boxes on images.
Handles drawing detections with various styles and options.
"""

import cv2
import numpy as np
from typing import Optional


def draw_boxes(
    image: np.ndarray, 
    boxes: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    label_size: int = 1,
    show_confidence: bool = False
) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image (np.ndarray): Input image (H x W x C) in BGR format
        boxes (np.ndarray): Bounding boxes (N x 5) - [x1, y1, x2, y2, confidence]
        color (tuple): Box color in BGR format (default: green)
        thickness (int): Line thickness for boxes (default: 2)
        label_size (int): Font size for labels (default: 1)
        show_confidence (bool): Whether to show confidence scores (default: False)
    
    Returns:
        np.ndarray: Image with drawn boxes
    
    Example:
        output_image = draw_boxes(image, boxes, color=(0, 255, 0), thickness=2)
    """
    if isinstance(image, np.ndarray):
        result_image = image.copy()
    else:
        result_image = np.array(image)
    
    # Ensure image is in BGR format (OpenCV convention)
    if len(result_image.shape) == 2:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    
    if len(boxes) == 0:
        return result_image
    
    # Draw each bounding box
    for box in boxes:
        x1, y1, x2, y2, confidence = box[:5]
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with confidence if requested
        if show_confidence:
            label = f"{confidence:.2f}"
            font_scale = label_size * 0.5
            font_thickness = max(1, int(label_size * 1.5))
            
            # Get text size for background rectangle
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Draw background rectangle for text
            text_x = x1
            text_y = max(15, y1 - 5)
            cv2.rectangle(
                result_image,
                (text_x, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y),
                color,
                -1
            )
            
            # Put text
            cv2.putText(
                result_image,
                label,
                (text_x + 2, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness
            )
        else:
            # Simple marker on top-left corner
            cv2.circle(result_image, (x1, y1), 3, (0, 0, 255), -1)  # Red circle
    
    return result_image


def add_count_text(
    image: np.ndarray,
    count: int,
    position: str = "top"
) -> np.ndarray:
    """
    Add tree count text to the image.
    
    Args:
        image (np.ndarray): Input image
        count (int): Number of trees detected
        position (str): Position of text - "top", "bottom", "top-left", etc.
    
    Returns:
        np.ndarray: Image with count text
    
    Example:
        output = add_count_text(image, count=42, position="top")
    """
    result_image = image.copy()
    height, width = result_image.shape[:2]
    
    text = f"Total Trees: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    font_color = (0, 255, 0)  # Green
    bg_color = (0, 0, 0)      # Black background
    
    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size
    padding = 10
    
    # Determine position
    if position == "top":
        text_x = (width - text_width) // 2
        text_y = text_height + padding
    elif position == "bottom":
        text_x = (width - text_width) // 2
        text_y = height - padding
    elif position == "top-left":
        text_x = padding
        text_y = text_height + padding
    elif position == "top-right":
        text_x = width - text_width - padding
        text_y = text_height + padding
    elif position == "bottom-left":
        text_x = padding
        text_y = height - padding
    elif position == "bottom-right":
        text_x = width - text_width - padding
        text_y = height - padding
    else:
        text_x = (width - text_width) // 2
        text_y = text_height + padding
    
    # Draw background rectangle
    cv2.rectangle(
        result_image,
        (text_x - padding, text_y - text_height - padding),
        (text_x + text_width + padding, text_y + padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        result_image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        font_color,
        font_thickness
    )
    
    return result_image


def create_grid_display(
    images: list,
    titles: list = None,
    figsize: tuple = (12, 8)
) -> None:
    """
    Display multiple images in a grid using matplotlib.
    Useful for debugging and visualization.
    
    Args:
        images (list): List of numpy arrays (images)
        titles (list): List of titles for each image
        figsize (tuple): Figure size for matplotlib
    
    Example:
        import matplotlib.pyplot as plt
        images = [original, processed, output]
        titles = ["Original", "Processed", "Output"]
        create_grid_display(images, titles)
        plt.show()
    """
    try:
        import matplotlib.pyplot as plt
        
        n_images = len(images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, img in enumerate(images):
            ax = axes[idx]
            
            # Convert BGR to RGB for display
            if len(img.shape) == 3 and img.shape[2] == 3:
                display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                display_img = img
            
            ax.imshow(display_img, cmap='gray' if len(display_img.shape) == 2 else None)
            
            if titles and idx < len(titles):
                ax.set_title(titles[idx])
            
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
    except ImportError:
        print("Warning: matplotlib not available for grid display")

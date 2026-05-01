"""
🌴 ULTRA-AGGRESSIVE COCONUT TREE DETECTION v3.0 🌴
Organized Project Structure
"""

import os
import sys
import cv2
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed")
    sys.exit(1)

from utils.tiling import tile_image, convert_tile_boxes_to_image_coords, clip_boxes_to_image
from utils.nms import apply_nms, merge_detections
from utils.draw_boxes import draw_boxes, add_count_text
from utils.ultra_aggressive import super_aggressive_detection


class Config:
    """Configuration for test2.png detection."""
    # Get the root directory (parent of src)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    INPUT_DIR = os.path.join(ROOT_DIR, "data", "input")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "detected")
    MODEL_DIR = os.path.join(ROOT_DIR, "data", "models")
    MODEL_NAME = "palm.pt"
    INPUT_IMAGE = "test2.png"  # ← Use test2.png
    OUTPUT_IMAGE = "detected.png"
    OUTPUT_COUNT_FILE = "count.txt"
    
    YOLO_CONFIDENCE = 0.0001
    TILE_SIZE = 512
    TILING_THRESHOLD = 1024
    IOU_THRESHOLD = 0.15


def detect_on_tile_ultimate(model, tile: np.ndarray) -> np.ndarray:
    """Ultimate tile detection."""
    all_detections = []
    
    try:
        results = model(tile, conf=Config.YOLO_CONFIDENCE, verbose=False)
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                yolo_det = np.column_stack([xyxy, conf * 0.8])
                all_detections.append(yolo_det)
    except:
        pass
    
    try:
        aggressive_det = super_aggressive_detection(tile)
        if len(aggressive_det) > 0:
            all_detections.append(aggressive_det)
    except:
        pass
    
    if len(all_detections) > 0:
        combined = np.vstack(all_detections)
        return combined
    return np.array([])


def main():
    """Main detection."""
    print("\n")
    print("╔════════════════════════════════════════════════════════╗")
    print("║ 🌴 ULTRA-AGGRESSIVE COCONUT DETECTION v3.0 🌴        ║")
    print("║    Processing: test2.png → results/detected/          ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    try:
        # Validate paths
        print("\n📋 Validating paths...")
        if not os.path.exists(Config.INPUT_DIR):
            print(f"❌ Input dir not found!")
            return False
        
        input_path = os.path.join(Config.INPUT_DIR, Config.INPUT_IMAGE)
        if not os.path.exists(input_path):
            print(f"❌ {Config.INPUT_IMAGE} not found in input/")
            print(f"   Available files:")
            for f in os.listdir(Config.INPUT_DIR):
                print(f"   - {f}")
            return False
        
        model_path = os.path.join(Config.MODEL_DIR, Config.MODEL_NAME)
        if not os.path.exists(model_path):
            print(f"❌ Model not found!")
            return False
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        print(f"✓ All paths verified!")
        print(f"✓ Output directory: {Config.OUTPUT_DIR}/")
        
        # Load model
        print(f"\n📦 Loading YOLO model...")
        model = YOLO(model_path)
        print(f"✓ Model ready!")
        
        # Load image
        print(f"\n🖼️  Loading image: {Config.INPUT_IMAGE}")
        image = cv2.imread(input_path)
        if image is None:
            print(f"❌ Failed to load image!")
            return False
        
        h, w = image.shape[:2]
        print(f"✓ Image: {w}x{h} pixels")
        
        # Detection
        print(f"\n🔍 Starting ULTIMATE detection...")
        print(f"   Methods: YOLO + Contours + Watershed + Edges + Green Channel")
        
        use_tiling = max(h, w) > Config.TILING_THRESHOLD
        
        if use_tiling:
            print(f"\n   Tiling enabled (tile size: {Config.TILE_SIZE}x{Config.TILE_SIZE})")
            tiles = tile_image(image, tile_size=Config.TILE_SIZE)
            print(f"   Created {len(tiles)} tiles")
            
            all_detections = []
            print(f"\n   Processing tiles:")
            
            for tile, tx, ty in tqdm(tiles, desc="   Progress"):
                tile_det = detect_on_tile_ultimate(model, tile)
                if len(tile_det) > 0:
                    img_det = convert_tile_boxes_to_image_coords(tile_det, tx, ty)
                    all_detections.append(img_det)
            
            if len(all_detections) > 0:
                merged = merge_detections(all_detections)
                merged = clip_boxes_to_image(merged, h, w)
            else:
                merged = np.array([])
        else:
            print(f"\n   Direct processing (no tiling)")
            merged = detect_on_tile_ultimate(model, image)
        
        # NMS
        print(f"\n   Raw detections: {len(merged)}")
        if len(merged) > 0:
            print(f"   Applying NMS (IoU: {Config.IOU_THRESHOLD})...")
            final = apply_nms(merged, iou_threshold=Config.IOU_THRESHOLD)
            print(f"   ✓ After NMS: {len(final)}")
        else:
            final = np.array([])
        
        count = len(final)
        
        # Draw and Save
        print(f"\n🎨 Drawing {count} detections...")
        result = draw_boxes(image, final, color=(0, 255, 0), thickness=2)
        result = add_count_text(result, count, position="top")
        
        print(f"\n💾 Saving results to {Config.OUTPUT_DIR}/")
        out_img = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_IMAGE)
        cv2.imwrite(out_img, result)
        print(f"✓ Image saved: {out_img}")
        
        out_txt = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_COUNT_FILE)
        with open(out_txt, 'w') as f:
            f.write(f"Coconut Tree Detection Results (test2.png)\n")
            f.write(f"==========================================\n\n")
            f.write(f"Total Trees Detected: {count}\n")
            f.write(f"Image: {Config.INPUT_IMAGE}\n")
            f.write(f"Detection Methods: YOLO + Contours + Watershed + Edges + Green\n")
        print(f"✓ Count saved: {out_txt}")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ DETECTION COMPLETE!")
        print("=" * 60)
        print(f"\n🌴 Trees Detected in {Config.INPUT_IMAGE}: {count}")
        print(f"\n📁 Output Folder: {Config.OUTPUT_DIR}/")
        print(f"   - {Config.OUTPUT_IMAGE} (with green bounding boxes)")
        print(f"   - {Config.OUTPUT_COUNT_FILE} (tree count)")
        print("\n" + "=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

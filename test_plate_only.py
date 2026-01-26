"""
Quick test script to run plate detection model in isolation.
Usage: python test_plate_only.py --image path/to/image.jpg
"""

import argparse
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Test plate detection model')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--model', type=str, default='models/plate_model.pt', help='Path to plate model')
    parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold (lower = more detections)')
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print("PLATE DETECTION TEST (ISOLATED)")
    print(f"{'='*50}")
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print(f"{'='*50}\n")

    # Load model
    model = YOLO(args.model)
    print(f"Model classes: {model.names}")

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Could not load image: {args.image}")
        return

    print(f"Image size: {img.shape[1]}x{img.shape[0]} pixels")

    # Run detection on FULL image
    print("\n--- Testing on FULL image ---")
    results = model(args.image, device='cpu', conf=args.conf, verbose=True)

    # Print all detections
    for r in results:
        boxes = r.boxes
        print(f"Total detections: {len(boxes)}")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            cls = int(box.cls)
            w, h = x2 - x1, y2 - y1
            print(f"  Detection {i+1}: class={cls}, conf={conf:.3f}, box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}], size={w:.0f}x{h:.0f}px")

    # Save annotated result
    results[0].save('plate_test_result.jpg')
    print(f"\nAnnotated image saved to: plate_test_result.jpg")

    # Also test with lower confidence to see if model detects ANYTHING
    if len(results[0].boxes) == 0:
        print("\n--- No detections! Trying with conf=0.01 ---")
        results_low = model(args.image, device='cpu', conf=0.01)
        for r in results_low:
            boxes = r.boxes
            print(f"Detections at conf=0.01: {len(boxes)}")
            for i, box in enumerate(boxes):
                conf = float(box.conf)
                cls = int(box.cls)
                print(f"  Detection {i+1}: class={cls}, conf={conf:.3f}")

    print(f"\n{'='*50}")
    print("TEST COMPLETE")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    main()

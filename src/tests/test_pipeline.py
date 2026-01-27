"""
Debug script to trace exactly what's happening in the pipeline.
Usage: python debug_pipeline.py --image "test picture 4.jpg"
"""

import argparse
import cv2
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--helmet-model', type=str, default='models/helmet_model.pt')
    parser.add_argument('--plate-model', type=str, default='models/plate_model.pt')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("PIPELINE DEBUG")
    print(f"{'='*60}")

    # Load models
    helmet_model = YOLO(args.helmet_model)
    plate_model = YOLO(args.plate_model)

    # Load image
    image = cv2.imread(args.image)
    print(f"\nImage: {args.image}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")

    # STEP 1: Helmet Detection
    print(f"\n{'='*60}")
    print("STEP 1: HELMET DETECTION")
    print(f"{'='*60}")

    helmet_results = helmet_model(args.image, device='cpu', conf=0.25)

    people = []
    for r in helmet_results:
        for box in r.boxes:
            person = {
                'helmet_status': int(box.cls),
                'detection_confidence': float(box.conf),
                'bounding_box_coordinates': box.xyxy[0].tolist()
            }
            people.append(person)

            x1, y1, x2, y2 = person['bounding_box_coordinates']
            w, h = x2 - x1, y2 - y1
            cls_name = helmet_model.names[person['helmet_status']]
            print(f"  Person: class={cls_name}, conf={person['detection_confidence']:.2f}")
            print(f"          box=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}], size={w:.0f}x{h:.0f}px")

    print(f"\nTotal people found: {len(people)}")

    if not people:
        print("No people detected! Stopping.")
        return

    # STEP 2: Two-Stage Plate Detection
    print(f"\n{'='*60}")
    print("STEP 2: TWO-STAGE PLATE DETECTION")
    print(f"{'='*60}")

    img_height, img_width = image.shape[:2]
    expand_ratio = 1.5
    extend_below = 2.0

    all_plates = []

    for i, person in enumerate(people):
        print(f"\n  --- Processing Person {i+1} ---")

        person_box = person['bounding_box_coordinates']
        px1, py1, px2, py2 = person_box

        person_width = px2 - px1
        person_height = py2 - py1

        print(f"  Person box: [{px1:.0f}, {py1:.0f}, {px2:.0f}, {py2:.0f}]")
        print(f"  Person size: {person_width:.0f}x{person_height:.0f}px")

        # Calculate crop region
        expand_x = person_width * (expand_ratio - 1) / 2
        crop_x1 = max(0, int(px1 - expand_x))
        crop_x2 = min(img_width, int(px2 + expand_x))
        crop_y1 = max(0, int(py1 - person_height * 0.2))
        crop_y2 = min(img_height, int(py2 + person_height * extend_below))

        print(f"  Crop region: [{crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}]")
        print(f"  Crop size: {crop_x2-crop_x1}x{crop_y2-crop_y1}px")

        # Actually crop
        cropped_region = image[crop_y1:crop_y2, crop_x1:crop_x2]
        print(f"  Cropped array shape: {cropped_region.shape}")

        # Save the crop for inspection
        crop_filename = f"src/tests/results/debug_crop_person{i+1}.jpg"
        cv2.imwrite(crop_filename, cropped_region)
        print(f"  Saved crop to: {crop_filename}")

        # Run plate detection on crop
        print(f"\n  Running plate detection on crop...")
        plate_results = plate_model(cropped_region, device='cpu', conf=0.1, verbose=False)

        crop_plates = []
        for r in plate_results:
            print(f"  Detections in crop: {len(r.boxes)}")
            for box in r.boxes:
                plate_coords = box.xyxy[0].tolist()
                conf = float(box.conf)
                print(f"    Plate (in crop coords): [{plate_coords[0]:.0f}, {plate_coords[1]:.0f}, {plate_coords[2]:.0f}, {plate_coords[3]:.0f}], conf={conf:.2f}")

                # Map back to original image coordinates
                orig_x1 = plate_coords[0] + crop_x1
                orig_y1 = plate_coords[1] + crop_y1
                orig_x2 = plate_coords[2] + crop_x1
                orig_y2 = plate_coords[3] + crop_y1

                print(f"    Plate (in original coords): [{orig_x1:.0f}, {orig_y1:.0f}, {orig_x2:.0f}, {orig_y2:.0f}]")

                all_plates.append([orig_x1, orig_y1, orig_x2, orig_y2])

    print(f"\n{'='*60}")
    print(f"TOTAL PLATES FOUND: {len(all_plates)}")
    print(f"{'='*60}")

    # STEP 3: Also test full image detection
    print(f"\n{'='*60}")
    print("COMPARISON: FULL IMAGE PLATE DETECTION")
    print(f"{'='*60}")

    full_results = plate_model(args.image, device='cpu', conf=0.1)
    for r in full_results:
        print(f"Plates in full image: {len(r.boxes)}")
        for box in r.boxes:
            coords = box.xyxy[0].tolist()
            conf = float(box.conf)
            print(f"  Plate: [{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}], conf={conf:.2f}")

    print(f"\n{'='*60}")
    print("DEBUG COMPLETE")
    print(f"{'='*60}")
    print("\nCheck the debug_crop_person*.jpg files to see what regions are being sent to the plate detector.")

if __name__ == '__main__':
    main()

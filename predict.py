"""
Prediction CLI for helmet violation detection.
Usage:
    python predict.py --image data/test.jpg
    python predict.py --folder data/test_images/
"""

import argparse
import cv2
import os
from pathlib import Path
from src.core import HelmetDetector, PlateDetector, PlateReader, SpatialMatcher
from src.utils import ReportGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect helmet violations and read license plates'
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image',
        type=str,
        help='Path to single image'
    )
    input_group.add_argument(
        '--folder',
        type=str,
        help='Path to folder containing images'
    )
    
    # Models
    parser.add_argument(
        '--helmet-model',
        type=str,
        default='models/helmet_model.pt',
        help='Path to helmet detection model'
    )
    parser.add_argument(
        '--plate-model',
        type=str,
        default='models/plate_model.pt',
        help='Path to plate detection model'
    )
    
    # Options
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for inference'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        default='csv',
        choices=['csv', 'json', 'excel'],
        help='Report output format'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Minimum detection confidence (0-1)'
    )
    
    return parser.parse_args()


def process_single_image(
    image_path: str,
    helmet_detector: HelmetDetector,
    plate_detector: PlateDetector,
    plate_reader: PlateReader,
    spatial_matcher: SpatialMatcher
) -> list:
    """Process a single image and return detections."""
    
    print(f"\n Processing: {os.path.basename(image_path)}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return []
    
    # Step 1: Detect all people (with/without helmets)
    people = helmet_detector.detect_helmets_in_image(image_path)
    print(f"Found {len(people)} people")
    
    if not people:
        return []
    
    # Step 2: Detect all license plates in the image
    all_plates = plate_detector.detect_all_plates(image)
    print(f"Found {len(all_plates)} license plates")
    
    # Step 3: Match each person to their plate using spatial matching
    matched_detections = spatial_matcher.match_people_to_plates(people, all_plates)
    
    # Step 4: Read text from matched plates
    results = []
    violations = 0
    
    for idx, person in enumerate(matched_detections, 1):
        has_helmet = helmet_detector.person_has_helmet(person['helmet_status'])
        
        plate_text = "NO_PLATE_MATCHED"
        plate_confidence = 0.0
        
        if person['plate_matched']:
            plate_text, plate_confidence = plate_reader.read_text_from_plate(
                image,
                person['matched_plate']
            )
        
        # Count violations
        if not has_helmet:
            violations += 1
            print(f"Person #{idx}: VIOLATION (no helmet) - Plate: {plate_text}")
        else:
            print(f"Person #{idx}: Compliant (helmet) - Plate: {plate_text}")
        
        results.append({
            'image_file': os.path.basename(image_path),
            'person_id': idx,
            'has_helmet': has_helmet,
            'helmet_status': helmet_detector.get_class_name(person['helmet_status']),
            'detection_confidence': person['detection_confidence'],
            'license_plate': plate_text,
            'plate_confidence': plate_confidence,
            'plate_matched': person['plate_matched']
        })
    
    print(f"Summary: {violations} violations, {len(people) - violations} compliant")
    
    return results


def main():
    """Main prediction function."""
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("  HELMET VIOLATION DETECTION - PREDICTION MODE")
    print("="*70)
    
    # Initialize all components
    print("\nInitializing models...")
    
    helmet_detector = HelmetDetector(
        args.helmet_model,
        device=args.device,
        confidence_threshold=args.confidence
    )
    
    plate_detector = PlateDetector(
        args.plate_model,
        device=args.device,
        confidence_threshold=args.confidence
    )
    
    plate_reader = PlateReader(
        supported_languages=['en'],
        gpu=(args.device == 'cuda')
    )
    
    spatial_matcher = SpatialMatcher(
        horizontal_threshold=200,
        vertical_overlap_threshold=50
    )
    
    report_generator = ReportGenerator()
    
    print("Models loaded successfully")
    
    # Process images
    all_results = []
    
    if args.image:
        # Single image mode
        results = process_single_image(
            args.image,
            helmet_detector,
            plate_detector,
            plate_reader,
            spatial_matcher
        )
        all_results.extend(results)
    
    elif args.folder:
        # Folder mode
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            f for f in os.listdir(args.folder)
            if any(f.lower().endswith(ext) for ext in image_extensions)
        ]
        
        print(f"\nProcessing {len(image_files)} images from folder...")
        
        for image_file in image_files:
            image_path = os.path.join(args.folder, image_file)
            results = process_single_image(
                image_path,
                helmet_detector,
                plate_detector,
                plate_reader,
                spatial_matcher
            )
            all_results.extend(results)
    
    # Generate report
    if all_results:
        report_generator.print_summary(all_results)
        report_path = report_generator.create_report(all_results, args.output_format)
        print(f"\nReport saved: {report_path}")
    else:
        print("\nNo detections found")
    
    print("\n" + "="*70)
    print("  PROCESSING COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

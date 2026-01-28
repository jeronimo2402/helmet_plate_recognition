"""
Prediction CLI for helmet violation detection.
Usage:
    python predict.py --image data/test.jpg
    python predict.py --folder data/test_images/
"""

import argparse
import os
from pathlib import Path
from src.core import HelmetDetector, PlateDetector, PlateReader, SpatialMatcher, ImageProcessor
from src.utils import ReportGenerator, ImageAnnotator


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
    parser.add_argument(
        '--save-images',
        action='store_true',
        default=True,
        help='Save annotated images showing violations and plate matches (default: enabled)'
    )
    parser.add_argument(
        '--no-save-images',
        action='store_false',
        dest='save_images',
        help='Disable saving annotated images'
    )

    return parser.parse_args()


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
    
    image_processor = ImageProcessor(
        helmet_detector,
        plate_detector,
        plate_reader,
        spatial_matcher
    )
    
    report_generator = ReportGenerator()

    image_annotator = None
    if args.save_images:
        image_annotator = ImageAnnotator()
        print("Image annotation enabled")

    print("Models loaded successfully")
    
    # Process images
    all_results = []
    
    if args.image:
        # Single image mode
        results = image_processor.process_single_image(args.image)

        # Save annotated image if enabled and add path to results
        annotated_path = None
        if image_annotator and results:
            annotated_path = image_annotator.annotate_and_save(
                args.image, results, helmet_detector
            )
            if annotated_path:
                print(f"Annotated image saved: {annotated_path}")

        # Add annotated image path to each result
        for result in results:
            result['annotated_image'] = annotated_path if annotated_path else ''

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
            results = image_processor.process_single_image(image_path)

            # Save annotated image if enabled and add path to results
            annotated_path = None
            if image_annotator and results:
                annotated_path = image_annotator.annotate_and_save(
                    image_path, results, helmet_detector
                )
                if annotated_path:
                    print(f"Annotated image saved: {annotated_path}")

            # Add annotated image path to each result
            for result in results:
                result['annotated_image'] = annotated_path if annotated_path else ''

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

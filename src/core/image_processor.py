"""
Shared image processing logic for helmet and plate detection.
Used by both predict.py and app.py to avoid code duplication.
"""

import cv2
import os
from typing import List, Dict, Optional


class ImageProcessor:
    """Processes images for helmet violation detection with plate reading."""
    
    def __init__(
        self,
        helmet_detector,
        plate_detector,
        plate_reader,
        spatial_matcher
    ):
        """
        Initialize the image processor with all required components.
        
        Args:
            helmet_detector: HelmetDetector instance
            plate_detector: PlateDetector instance
            plate_reader: PlateReader instance
            spatial_matcher: SpatialMatcher instance
        """
        self.helmet_detector = helmet_detector
        self.plate_detector = plate_detector
        self.plate_reader = plate_reader
        self.spatial_matcher = spatial_matcher
    
    def process_single_image(
        self,
        image_path: str,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Process a single image and return detection results.
        
        Args:
            image_path: Path to the image file
            verbose: Whether to print progress messages
            
        Returns:
            List of detection results, one dict per person detected
        """
        if verbose:
            print(f"\nProcessing: {os.path.basename(image_path)}")
        
        image = cv2.imread(image_path)
        if image is None:
            if verbose:
                print(f"Could not load image: {image_path}")
            return []
        
        people = self.helmet_detector.detect_helmets_in_image(image_path)
        if verbose:
            print(f"Found {len(people)} people")
        
        if not people:
            return []
        
        full_image_plates = self.plate_detector.detect_all_plates(image)
        
        cropped_region_plates = self.plate_detector.detect_plates_around_people(
            image, people, expand_ratio=1.5, extend_below=2.0
        )
        
        all_plates = full_image_plates + cropped_region_plates
        
        unique_plates = self._remove_duplicate_plates(all_plates, tolerance=20)
        
        if verbose:
            print(f"Found {len(unique_plates)} license plates (full: {len(full_image_plates)}, cropped: {len(cropped_region_plates)})")
        
        matched_detections = self.spatial_matcher.match_people_to_plates(people, unique_plates)
        
        results = self._build_results(
            image, 
            matched_detections, 
            image_path, 
            verbose
        )
        
        return results
    
    def _remove_duplicate_plates(
        self, 
        plates: List[List[float]], 
        tolerance: int = 20
    ) -> List[List[float]]:
        """
        Remove duplicate plate detections within tolerance.
        
        Args:
            plates: List of plate bounding boxes
            tolerance: Pixel tolerance for considering plates as duplicates
            
        Returns:
            List of unique plate bounding boxes
        """
        unique_plates = []
        for plate in plates:
            is_duplicate = False
            for existing in unique_plates:
                if (abs(plate[0] - existing[0]) < tolerance and 
                    abs(plate[1] - existing[1]) < tolerance and
                    abs(plate[2] - existing[2]) < tolerance and
                    abs(plate[3] - existing[3]) < tolerance):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_plates.append(plate)
        
        return unique_plates
    
    def _build_results(
        self,
        image,
        matched_detections: List[Dict],
        image_path: str,
        verbose: bool
    ) -> List[Dict]:
        """
        Build result dictionaries from matched detections.
        
        Args:
            image: Loaded image (numpy array)
            matched_detections: List of people with matched plates
            image_path: Path to the image file
            verbose: Whether to print progress
            
        Returns:
            List of result dictionaries
        """
        results = []
        violations = 0
        
        for idx, person in enumerate(matched_detections, 1):
            has_helmet = self.helmet_detector.person_has_helmet(person['helmet_status'])
            
            plate_text = "NO_PLATE_MATCHED"
            plate_confidence = 0.0
            
            if person['plate_matched']:
                plate_text, plate_confidence = self.plate_reader.read_text_from_plate(
                    image,
                    person['matched_plate']
                )
            
            if not has_helmet:
                violations += 1
                if verbose:
                    print(f"Person #{idx}: VIOLATION (no helmet) - Plate: {plate_text}")
            else:
                if verbose:
                    print(f"Person #{idx}: Compliant (helmet) - Plate: {plate_text}")
            
            results.append({
                'image_file': os.path.basename(image_path),
                'person_id': idx,
                'has_helmet': has_helmet,
                'helmet_status': self.helmet_detector.get_class_name(person['helmet_status']),
                'detection_confidence': person['detection_confidence'],
                'license_plate': plate_text,
                'plate_confidence': plate_confidence,
                'plate_matched': person['plate_matched'],
                'person_box': person['bounding_box_coordinates'],
                'matched_plate_box': person['matched_plate'] if person['plate_matched'] else None
            })
        
        if verbose:
            print(f"Summary: {violations} violations, {len(matched_detections) - violations} compliant")
        
        return results

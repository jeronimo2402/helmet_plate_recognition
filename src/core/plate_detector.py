"""
License plate detection module using YOLOv8.
Detects license plates in images or image regions.
"""

from ultralytics import YOLO
import numpy as np
from typing import List, Optional
import os


class PlateDetector:
    """Detects license plates in images using trained YOLO model."""
    
    PLATE_CLASS = 0  # License plates are class 0 in our trained model
    
    def __init__(
        self, 
        path_to_model: str = 'models/plate_model.pt',
        device: str = 'cpu',
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the plate detector.
        
        Args:
            path_to_model: Path to trained YOLO model for plates
            device: Device to run inference on ('cpu' or 'cuda')
            confidence_threshold: Minimum confidence for detections
        """
        if not os.path.exists(path_to_model):
            raise FileNotFoundError(f"Plate model not found: {path_to_model}")
        
        self.plate_detection_model = YOLO(path_to_model)
        self.device = device
        self.confidence_threshold = confidence_threshold
    
    def find_license_plates(
        self, 
        image_array: np.ndarray, 
        person_bounding_box: Optional[List[float]] = None
    ) -> List[List[float]]:
        """
        Detect license plates in an image or image region.
        
        Args:
            image_array: Image as numpy array (BGR format)
            person_bounding_box: Optional [x1, y1, x2, y2] to search only near person
            
        Returns:
            List of plate bounding boxes [[x1, y1, x2, y2], ...]
        """
        # For best results, search entire image (not just person region)
        # Spatial matching will handle person-to-plate association
        detection_results = self.plate_detection_model(
            image_array,
            device=self.device,
            conf=self.confidence_threshold
        )
        
        list_of_plate_locations = []
        
        for single_result in detection_results:
            for detected_box in single_result.boxes:
                object_class = int(detected_box.cls)
                confidence = float(detected_box.conf)
                
                # Check if this is a plate and meets confidence threshold
                if object_class == self.PLATE_CLASS and confidence >= self.confidence_threshold:
                    plate_coordinates = detected_box.xyxy[0].tolist()
                    list_of_plate_locations.append(plate_coordinates)
        
        return list_of_plate_locations
    
    def detect_all_plates(self, image_array: np.ndarray) -> List[List[float]]:
        """
        Detect all plates in full image (convenience method).

        Args:
            image_array: Full image as numpy array

        Returns:
            List of all detected plate bounding boxes
        """
        return self.find_license_plates(image_array, person_bounding_box=None)

    def detect_plates_around_people(
        self,
        image_array: np.ndarray,
        people_boxes: List[dict],
        expand_ratio: float = 1.5,
        extend_below: float = 2.0
    ) -> List[List[float]]:
        """
        Two-stage detection: crop around each person, detect plates in crops,
        then map coordinates back to original image.

        This helps detect small plates by effectively "zooming in" on each person.

        Args:
            image_array: Full image as numpy array (BGR format)
            people_boxes: List of person dicts with 'bounding_box_coordinates'
            expand_ratio: How much to expand crop horizontally (1.5 = 50% wider)
            extend_below: How much to extend below person to catch plate (2.0 = 2x person height)

        Returns:
            List of plate bounding boxes in ORIGINAL image coordinates
        """
        img_height, img_width = image_array.shape[:2]
        all_plates = []
        seen_plates = set()  # Avoid duplicates from overlapping crops

        for person in people_boxes:
            person_box = person['bounding_box_coordinates']
            px1, py1, px2, py2 = person_box

            # Calculate person dimensions
            person_width = px2 - px1
            person_height = py2 - py1

            # Create expanded crop region
            # Expand horizontally
            expand_x = person_width * (expand_ratio - 1) / 2
            crop_x1 = max(0, int(px1 - expand_x))
            crop_x2 = min(img_width, int(px2 + expand_x))

            # Extend below to catch the plate (bikes have plates below the rider)
            crop_y1 = max(0, int(py1 - person_height * 0.2))  # Slight extension above
            crop_y2 = min(img_height, int(py2 + person_height * extend_below))

            # Crop the region
            cropped_region = image_array[crop_y1:crop_y2, crop_x1:crop_x2]

            # Skip if crop is too small
            if cropped_region.shape[0] < 20 or cropped_region.shape[1] < 20:
                continue

            # Detect plates in cropped region
            crop_plates = self.find_license_plates(cropped_region)

            # Map coordinates back to original image
            for plate in crop_plates:
                # plate is [x1, y1, x2, y2] in crop coordinates
                orig_x1 = plate[0] + crop_x1
                orig_y1 = plate[1] + crop_y1
                orig_x2 = plate[2] + crop_x1
                orig_y2 = plate[3] + crop_y1

                # Create a key to avoid duplicates (round to nearest 10 pixels)
                plate_key = (
                    round(orig_x1 / 10) * 10,
                    round(orig_y1 / 10) * 10,
                    round(orig_x2 / 10) * 10,
                    round(orig_y2 / 10) * 10
                )

                if plate_key not in seen_plates:
                    seen_plates.add(plate_key)
                    all_plates.append([orig_x1, orig_y1, orig_x2, orig_y2])

        return all_plates

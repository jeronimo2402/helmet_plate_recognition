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
        confidence_threshold: float = 0.25
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

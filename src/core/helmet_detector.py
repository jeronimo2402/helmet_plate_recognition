"""
Helmet detection module using YOLOv8.
Detects people with and without helmets in images.
"""

from ultralytics import YOLO
from typing import List, Dict
import os


class HelmetDetector:
    """Detects helmets and no-helmets in images using trained YOLO model."""
    
    HELMET_CLASS = 0
    NO_HELMET_CLASS = 1
    
    def __init__(
        self, 
        trained_model_path: str,
        device: str = 'cpu',
        confidence_threshold: float = 0.25
    ):
        """
        Initialize the helmet detector.
        
        Args:
            trained_model_path: Path to trained YOLO model (.pt file)
            device: Device to run inference on ('cpu' or 'cuda')
            confidence_threshold: Minimum confidence for detections
        """
        if not os.path.exists(trained_model_path):
            raise FileNotFoundError(f"Model not found: {trained_model_path}")
        
        self.detection_model = YOLO(trained_model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
    
    def detect_helmets_in_image(self, image_file_path: str) -> List[Dict]:
        """
        Detect all people (with/without helmets) in an image.
        
        Args:
            image_file_path: Path to image file
            
        Returns:
            List of detections, each containing:
                - helmet_status: 0 (helmet) or 1 (no-helmet)
                - detection_confidence: Confidence score (0-1)
                - bounding_box_coordinates: [x1, y1, x2, y2]
        """
        detection_results = self.detection_model(
            image_file_path, 
            device=self.device,
            conf=self.confidence_threshold
        )
        
        list_of_people_found = []
        
        for single_result in detection_results:
            for detected_box in single_result.boxes:
                confidence = float(detected_box.conf)
                
                # Additional confidence filtering
                if confidence < self.confidence_threshold:
                    continue
                
                person_info = {
                    'helmet_status': int(detected_box.cls),
                    'detection_confidence': confidence,
                    'bounding_box_coordinates': detected_box.xyxy[0].tolist()
                }
                list_of_people_found.append(person_info)
        
        return list_of_people_found
    
    def person_has_helmet(self, helmet_status: int) -> bool:
        """
        Check if a person is wearing a helmet.
        
        Args:
            helmet_status: Class ID from detection (0 or 1)
            
        Returns:
            True if wearing helmet, False otherwise
        """
        return helmet_status == self.HELMET_CLASS
    
    def get_class_name(self, helmet_status: int) -> str:
        """Get human-readable class name."""
        return 'helmet' if helmet_status == self.HELMET_CLASS else 'no-helmet'

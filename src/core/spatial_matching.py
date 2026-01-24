"""
Spatial matching module for pairing people with their license plates.
Implements intelligent matching based on proximity and spatial relationships.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class SpatialMatcher:
    """Matches people to their corresponding license plates using spatial relationships."""
    
    def __init__(
        self,
        horizontal_threshold: int = 200,
        vertical_overlap_threshold: int = 50
    ):
        """
        Initialize the spatial matcher.
        
        Args:
            horizontal_threshold: Maximum horizontal distance (pixels) to consider
            vertical_overlap_threshold: Allowed vertical overlap (pixels)
        """
        self.horizontal_threshold = horizontal_threshold
        self.vertical_overlap_threshold = vertical_overlap_threshold
    
    @staticmethod
    def calculate_distance(box1: List[float], box2: List[float]) -> float:
        """
        Calculate Euclidean distance between centers of two bounding boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            Distance between box centers in pixels
        """
        x1_center = (box1[0] + box1[2]) / 2
        y1_center = (box1[1] + box1[3]) / 2
        
        x2_center = (box2[0] + box2[2]) / 2
        y2_center = (box2[1] + box2[3]) / 2
        
        distance = np.sqrt((x2_center - x1_center)**2 + (y2_center - y1_center)**2)
        
        return float(distance)
    
    def is_plate_below_person(
        self,
        person_box: List[float],
        plate_box: List[float]
    ) -> bool:
        """
        Check if a license plate is positioned below a person.
        
        In typical motorcycle images, the plate is mounted below the rider.
        This ensures we don't match plates that are above or beside the person.
        
        Args:
            person_box: Person bounding box [x1, y1, x2, y2]
            plate_box: Plate bounding box [x1, y1, x2, y2]
            
        Returns:
            True if plate is below person, False otherwise
        """
        person_bottom = person_box[3]  # y2 of person
        plate_top = plate_box[1]  # y1 of plate
        
        # Plate top should be at or below person bottom (with small overlap allowed)
        return plate_top > (person_bottom - self.vertical_overlap_threshold)
    
    def is_horizontally_aligned(
        self,
        person_box: List[float],
        plate_box: List[float]
    ) -> bool:
        """
        Check if a plate is horizontally aligned with a person.
        
        The plate should be roughly under the person, not far to the left or right.
        
        Args:
            person_box: Person bounding box [x1, y1, x2, y2]
            plate_box: Plate bounding box [x1, y1, x2, y2]
            
        Returns:
            True if horizontally aligned, False otherwise
        """
        person_x_center = (person_box[0] + person_box[2]) / 2
        plate_x_center = (plate_box[0] + plate_box[2]) / 2
        
        horizontal_distance = abs(plate_x_center - person_x_center)
        
        return horizontal_distance < self.horizontal_threshold
    
    def find_matching_plate(
        self,
        person_box: List[float],
        all_plates: List[List[float]]
    ) -> Optional[List[float]]:
        """
        Find the license plate that belongs to a specific person.
        
        Matching criteria:
        1. Plate must be BELOW the person (motorcycles have plates at bottom)
        2. Plate must be horizontally ALIGNED with the person
        3. From valid candidates, choose the CLOSEST plate
        
        Args:
            person_box: Bounding box of the person [x1, y1, x2, y2]
            all_plates: List of all detected plate bounding boxes
            
        Returns:
            Bounding box of matching plate, or None if no match found
        """
        valid_plates = []
        
        for plate_box in all_plates:
            # Test 1: Is plate below person?
            if not self.is_plate_below_person(person_box, plate_box):
                continue
            
            # Test 2: Is plate horizontally aligned?
            if not self.is_horizontally_aligned(person_box, plate_box):
                continue
            
            # Calculate distance for ranking
            distance = self.calculate_distance(person_box, plate_box)
            
            valid_plates.append({
                'box': plate_box,
                'distance': distance
            })
        
        # Return the closest valid plate
        if valid_plates:
            closest = min(valid_plates, key=lambda x: x['distance'])
            return closest['box']
        
        return None
    
    def match_people_to_plates(
        self,
        people_boxes: List[Dict],
        plate_boxes: List[List[float]]
    ) -> List[Dict]:
        """
        Match multiple people to their corresponding license plates.
        
        Args:
            people_boxes: List of dicts with person info including 'bounding_box'
            plate_boxes: List of all detected plate bounding boxes
            
        Returns:
            List of people with their matched plates added
        """
        results = []
        
        for person in people_boxes:
            person_box = person['bounding_box_coordinates']
            
            # Find matching plate for this person
            matched_plate = self.find_matching_plate(person_box, plate_boxes)
            
            # Add plate info to person record
            person_with_plate = person.copy()
            person_with_plate['matched_plate'] = matched_plate
            person_with_plate['plate_matched'] = matched_plate is not None
            
            results.append(person_with_plate)
        
        return results

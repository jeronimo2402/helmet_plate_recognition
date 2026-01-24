"""
License plate text reading module using EasyOCR.
Extracts text from cropped license plate images with preprocessing.
"""

import easyocr
import cv2
import numpy as np
from typing import List, Tuple, Optional


class PlateReader:
    """Reads text from license plate images using OCR."""
    
    def __init__(
        self, 
        supported_languages: List[str] = None,
        gpu: bool = False,
        min_confidence: float = 0.1
    ):
        """
        Initialize the plate text reader.
        
        Args:
            supported_languages: Languages for OCR (default: ['en'])
            gpu: Whether to use GPU for OCR
            min_confidence: Minimum OCR confidence to accept result
        """
        if supported_languages is None:
            supported_languages = ['en']
        
        self.text_recognition_engine = easyocr.Reader(supported_languages, gpu=gpu)
        self.min_confidence = min_confidence
    
    def _preprocess_plate_image(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image to improve OCR accuracy.
        
        Applies:
        - Grayscale conversion
        - Histogram equalization for contrast enhancement
        
        Args:
            plate_image: Cropped plate image in BGR
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        grayscale = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast with histogram equalization
        enhanced = cv2.equalizeHist(grayscale)
        
        return enhanced
    
    def read_text_from_plate(
        self, 
        full_image: np.ndarray, 
        plate_bounding_box: List[float],
        use_preprocessing: bool = True
    ) -> Tuple[str, float]:
        """
        Read text from a license plate region.
        
        Args:
            full_image: Full image as numpy array (BGR)
            plate_bounding_box: Plate location [x1, y1, x2, y2]
            use_preprocessing: Whether to apply preprocessing
            
        Returns:
            Tuple of (plate_text, confidence)
            Returns ("NO_TEXT_DETECTED", 0.0) if OCR fails
        """
        left_x, top_y, right_x, bottom_y = map(int, plate_bounding_box)
        
        # Validate coordinates
        if left_x >= right_x or top_y >= bottom_y:
            return "INVALID_COORDINATES", 0.0
        
        # Crop plate region
        cropped_plate_image = full_image[top_y:bottom_y, left_x:right_x]
        
        # Check if plate is too small
        height, width = cropped_plate_image.shape[:2]
        if width < 20 or height < 10:
            return "PLATE_TOO_SMALL", 0.0
        
        # Try OCR on original image first
        text_detection_results = self.text_recognition_engine.readtext(
            cropped_plate_image
        )
        
        # If failed and preprocessing enabled, try with preprocessing
        if not text_detection_results and use_preprocessing:
            preprocessed = self._preprocess_plate_image(cropped_plate_image)
            text_detection_results = self.text_recognition_engine.readtext(
                preprocessed
            )
        
        # Extract text and confidence
        if text_detection_results:
            # Sort by confidence and take best results
            sorted_results = sorted(
                text_detection_results, 
                key=lambda x: x[2], 
                reverse=True
            )
            
            # Extract text parts with sufficient confidence
            valid_texts = [
                result[1] for result in sorted_results 
                if result[2] >= self.min_confidence
            ]
            
            if valid_texts:
                final_plate_text = ' '.join(valid_texts)
                avg_confidence = np.mean([r[2] for r in sorted_results[:len(valid_texts)]])
                return final_plate_text, float(avg_confidence)
        
        return "NO_TEXT_DETECTED", 0.0
    
    def validate_plate_format(self, plate_text: str) -> bool:
        """
        Basic validation of plate text format.
        
        Args:
            plate_text: Extracted plate text
            
        Returns:
            True if format seems valid, False otherwise
        """
        if not plate_text or plate_text in ["NO_TEXT_DETECTED", "PLATE_TOO_SMALL"]:
            return False
        
        # Remove spaces and check if contains alphanumeric
        cleaned = plate_text.replace(" ", "")
        return len(cleaned) >= 3 and any(c.isalnum() for c in cleaned)

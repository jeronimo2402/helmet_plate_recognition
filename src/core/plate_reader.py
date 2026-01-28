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
        min_confidence: float = 0.25
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
    
    def _assess_plate_quality(self, plate_image: np.ndarray) -> dict:
        """
        Assess the quality of a plate image to determine OCR difficulty.
        
        Args:
            plate_image: Cropped plate image
            
        Returns:
            Dict with quality metrics
        """
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate contrast
        contrast = gray.std()
        
        # Calculate brightness
        brightness = gray.mean()
        
        # Determine quality level
        is_blurry = laplacian_var < 50
        is_low_contrast = contrast < 30
        is_too_dark = brightness < 50
        is_too_bright = brightness > 200
        
        quality_score = 0
        if not is_blurry:
            quality_score += 1
        if not is_low_contrast:
            quality_score += 1
        if not is_too_dark and not is_too_bright:
            quality_score += 1
        
        return {
            'blur_score': laplacian_var,
            'contrast': contrast,
            'brightness': brightness,
            'is_blurry': is_blurry,
            'is_low_contrast': is_low_contrast,
            'quality_score': quality_score,
            'is_good_quality': quality_score >= 2
        }
    
    def _upscale_plate(self, plate_image: np.ndarray, scale: int = 2) -> np.ndarray:
        """
        Upscale small plate images for better OCR.
        
        Args:
            plate_image: Plate image
            scale: Upscaling factor
            
        Returns:
            Upscaled image
        """
        height, width = plate_image.shape[:2]
        return cv2.resize(
            plate_image, 
            (width * scale, height * scale), 
            interpolation=cv2.INTER_CUBIC
        )
    
    def _preprocess_plate_image(self, plate_image: np.ndarray, method: str = 'enhanced') -> np.ndarray:
        """
        Preprocess plate image to improve OCR accuracy.
        
        Args:
            plate_image: Cropped plate image in BGR
            method: Preprocessing method
            
        Returns:
            Preprocessed image
        """
        # Upscale if plate is small (helps with low resolution)
        height, width = plate_image.shape[:2]
        if width < 100 or height < 40:
            plate_image = self._upscale_plate(plate_image, scale=2)
        
        # Convert to grayscale
        grayscale = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        if method == 'enhanced':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(grayscale)
        
        elif method == 'binary':
            # Adaptive thresholding with denoising
            denoised = cv2.fastNlMeansDenoising(grayscale, h=10)
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            return binary
        
        elif method == 'denoise':
            # Strong denoising + CLAHE
            denoised = cv2.fastNlMeansDenoising(grayscale, h=15)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(denoised)
        
        elif method == 'sharpen':
            # Sharpen to enhance text edges
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            sharpened = cv2.filter2D(grayscale, -1, kernel)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(sharpened)
        
        elif method == 'morph':
            # Morphological operations to clean up text
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(grayscale)
            
            # Apply morphological closing to connect broken characters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            return morph
        
        elif method == 'otsu':
            # Otsu's binarization (automatic threshold)
            denoised = cv2.fastNlMeansDenoising(grayscale, h=10)
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        
        return grayscale
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean up common OCR errors in license plate text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text or text in ["NO_TEXT_DETECTED", "PLATE_TOO_SMALL", "INVALID_COORDINATES"]:
            return text
        
        # Remove common noise characters
        cleaned = text.strip()
        
        # Remove special characters except hyphens and spaces
        import re
        cleaned = re.sub(r'[^A-Z0-9\s\-]', '', cleaned.upper())
        
        # Fix common OCR mistakes (context-aware)
        # These are common confusions in license plates
        replacements = {
            'O': '0',  # Letter O -> Number 0 in numeric contexts
            'I': '1',  # Letter I -> Number 1 in numeric contexts
            'S': '5',  # Letter S -> Number 5 in numeric contexts
            'Z': '2',  # Letter Z -> Number 2 in numeric contexts
            'B': '8',  # Letter B -> Number 8 in numeric contexts
        }
        
        # Apply replacements intelligently
        # If text has both letters and numbers, be more conservative
        parts = cleaned.split()
        cleaned_parts = []
        
        for part in parts:
            # If part is mostly numeric, apply number corrections
            digit_count = sum(c.isdigit() for c in part)
            alpha_count = sum(c.isalpha() for c in part)
            
            if digit_count > alpha_count:
                # Likely a number, fix letter confusions
                for old, new in replacements.items():
                    part = part.replace(old, new)
            
            cleaned_parts.append(part)
        
        cleaned = ' '.join(cleaned_parts)
        
        # Remove excessive spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
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
        if width < 30 or height < 15:
            return "PLATE_TOO_SMALL", 0.0
        
        # Assess plate quality to determine OCR strategy
        quality_info = self._assess_plate_quality(cropped_plate_image)
        
        # Try multiple preprocessing strategies and keep best result
        best_text = "NO_TEXT_DETECTED"
        best_confidence = 0.0
        
        # Adjust strategies based on quality
        if quality_info['is_good_quality']:
            # Good quality: try fewer strategies (faster)
            strategies = ['original', 'enhanced', 'sharpen']
        else:
            # Poor quality: try all strategies (more thorough)
            strategies = ['original']
            if use_preprocessing:
                # Prioritize strategies that help with specific issues
                if quality_info['is_blurry']:
                    strategies.extend(['sharpen', 'enhanced', 'morph', 'denoise', 'binary', 'otsu'])
                elif quality_info['is_low_contrast']:
                    strategies.extend(['enhanced', 'binary', 'otsu', 'sharpen', 'morph', 'denoise'])
                else:
                    strategies.extend(['enhanced', 'sharpen', 'binary', 'denoise', 'morph', 'otsu'])
        
        for strategy in strategies:
            if strategy == 'original':
                test_image = cropped_plate_image
            else:
                test_image = self._preprocess_plate_image(cropped_plate_image, method=strategy)
            
            # Run OCR
            text_detection_results = self.text_recognition_engine.readtext(test_image)
            
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
                    candidate_text = ' '.join(valid_texts)
                    candidate_confidence = np.mean([r[2] for r in sorted_results[:len(valid_texts)]])
                    
                    # Keep this result if it's better than previous attempts
                    if candidate_confidence > best_confidence:
                        best_text = candidate_text
                        best_confidence = float(candidate_confidence)
        
        # Clean up the best result before returning
        if best_text not in ["NO_TEXT_DETECTED", "PLATE_TOO_SMALL", "INVALID_COORDINATES"]:
            best_text = self._clean_ocr_text(best_text)
        
        return best_text, best_confidence
    
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

"""
Image annotation module for visualizing helmet violations and plate matches.
Draws bounding boxes, labels, and connecting lines on images.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
from datetime import datetime


class ImageAnnotator:
    """Annotates images with detection results and saves them."""

    # Color definitions (BGR format for OpenCV)
    COLOR_VIOLATION = (0, 0, 255)      # Red - no helmet
    COLOR_COMPLIANT = (0, 255, 0)      # Green - has helmet
    COLOR_PLATE = (0, 255, 255)        # Yellow - license plate
    COLOR_LINE = (0, 255, 255)         # Yellow - connecting line
    COLOR_UNREADABLE = (0, 165, 255)   # Orange - unreadable plate

    def __init__(self, output_folder: str = 'outputs/reports/images'):
        """
        Initialize the image annotator.

        Args:
            output_folder: Directory to save annotated images
        """
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def _get_box_center(self, box: List[float]) -> Tuple[int, int]:
        """Calculate the center point of a bounding box."""
        x_center = int((box[0] + box[2]) / 2)
        y_center = int((box[1] + box[3]) / 2)
        return (x_center, y_center)

    def _draw_box_with_label(
        self,
        image: np.ndarray,
        box: List[float],
        label: str,
        color: Tuple[int, int, int],
        thickness: int = 3,
        font_scale: float = 0.7
    ) -> None:
        """Draw a bounding box with a label on the image."""
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw label background for better visibility
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 5, y1),
            color,
            -1  # Filled
        )

        # Draw label text
        cv2.putText(
            image, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2
        )

    def _draw_plate_text(
        self,
        image: np.ndarray,
        box: List[float],
        plate_text: str,
        color: Tuple[int, int, int]
    ) -> None:
        """Draw plate text below the plate bounding box."""
        x1, y2 = int(box[0]), int(box[3])

        # Draw background for text
        (text_width, text_height), _ = cv2.getTextSize(
            plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            image,
            (x1, y2 + 5),
            (x1 + text_width + 10, y2 + text_height + 15),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            image, plate_text, (x1 + 5, y2 + text_height + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )

    def _draw_connecting_line(
        self,
        image: np.ndarray,
        person_box: List[float],
        plate_box: List[float],
        color: Tuple[int, int, int]
    ) -> None:
        """Draw a line connecting a person to their matched plate."""
        person_center = self._get_box_center(person_box)
        plate_center = self._get_box_center(plate_box)

        cv2.line(image, person_center, plate_center, color, 2, cv2.LINE_AA)

    def annotate_image(
        self,
        image_path: str,
        detections: List[Dict],
        helmet_detector
    ) -> np.ndarray:
        """
        Annotate an image with all detection results.

        Args:
            image_path: Path to the original image
            detections: List of detection results from process_single_image
            helmet_detector: HelmetDetector instance (for checking helmet status)

        Returns:
            Annotated image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        violation_count = 0

        for detection in detections:
            has_helmet = detection.get('has_helmet', False)
            person_box = detection.get('person_box')
            plate_box = detection.get('matched_plate_box')
            plate_text = detection.get('license_plate', 'NO_PLATE_MATCHED')
            plate_matched = detection.get('plate_matched', False)

            if person_box is None:
                continue

            if has_helmet:
                # Compliant - green box
                self._draw_box_with_label(
                    image, person_box, "HELMET", self.COLOR_COMPLIANT
                )
            else:
                # Violation - red box
                violation_count += 1
                self._draw_box_with_label(
                    image, person_box, f"NO HELMET #{violation_count}", self.COLOR_VIOLATION
                )

                # Draw plate if matched
                if plate_matched and plate_box is not None:
                    # Draw plate box
                    self._draw_box_with_label(
                        image, plate_box, "PLATE", self.COLOR_PLATE
                    )

                    # Draw connecting line
                    self._draw_connecting_line(
                        image, person_box, plate_box, self.COLOR_LINE
                    )

                    # Draw plate text
                    if plate_text and plate_text != 'NO_PLATE_MATCHED':
                        self._draw_plate_text(image, plate_box, plate_text, self.COLOR_PLATE)
                    else:
                        self._draw_plate_text(image, plate_box, "UNREADABLE", self.COLOR_UNREADABLE)

        return image

    def save_annotated_image(
        self,
        image: np.ndarray,
        original_filename: str,
        suffix: str = '_annotated'
    ) -> str:
        """
        Save an annotated image to the output folder.

        Args:
            image: Annotated image as numpy array
            original_filename: Original image filename (for naming)
            suffix: Suffix to add before extension

        Returns:
            Path to saved annotated image
        """
        # Create output filename
        name, ext = os.path.splitext(original_filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{name}{suffix}_{timestamp}{ext}"
        output_path = os.path.join(self.output_folder, output_filename)

        # Save image
        cv2.imwrite(output_path, image)

        return output_path

    def annotate_and_save(
        self,
        image_path: str,
        detections: List[Dict],
        helmet_detector
    ) -> Optional[str]:
        """
        Annotate an image and save it in one step.

        Args:
            image_path: Path to original image
            detections: List of detection results
            helmet_detector: HelmetDetector instance

        Returns:
            Path to saved annotated image, or None if no detections
        """
        if not detections:
            return None

        # Annotate
        annotated = self.annotate_image(image_path, detections, helmet_detector)

        # Save
        original_filename = os.path.basename(image_path)
        saved_path = self.save_annotated_image(annotated, original_filename)

        return saved_path

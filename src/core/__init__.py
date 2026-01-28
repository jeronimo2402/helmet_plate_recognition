"""Core detection modules for helmet and license plate recognition."""

from .helmet_detector import HelmetDetector
from .plate_detector import PlateDetector
from .plate_reader import PlateReader
from .spatial_matching import SpatialMatcher
from .image_processor import ImageProcessor

__all__ = [
    'HelmetDetector',
    'PlateDetector',
    'PlateReader',
    'SpatialMatcher',
    'ImageProcessor',
]

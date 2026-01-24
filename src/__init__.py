"""Helmet and License Plate Recognition System."""

# Export main modules for convenient imports
from .core import (
    HelmetDetector,
    PlateDetector,
    PlateReader,
    SpatialMatcher,
)
from .utils import ReportGenerator
from .training import DatasetDownloader, ModelTrainer

__version__ = "0.1.0"

__all__ = [
    # Core detection
    'HelmetDetector',
    'PlateDetector',
    'PlateReader',
    'SpatialMatcher',
    # Utils
    'ReportGenerator',
    # Training
    'DatasetDownloader',
    'ModelTrainer',
]

"""
Model training module using YOLOv8.
Handles training configuration and execution for both helmet and plate models.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

class ModelTrainer:
    """Trains YOLOv8 models for helmet and plate detection."""
    
    def __init__(
        self,
        base_model: str = 'yolov8n.pt',
        device: str = 'cpu',
        output_dir: str = os.getenv('RUNS_PATH')
    ):
        """
        Initialize the model trainer.
        
        Args:
            base_model: Pre-trained YOLO model to start from
            device: 'cpu' or 'cuda' for training
            output_dir: Directory to save training results
        """
        self.base_model = base_model
        self.device = device
        self.output_dir = output_dir
        
    def train_helmet_model(
        self,
        data_yaml_path: str,
        epochs: int = 50,
        image_size: int = 640,
        batch_size: int = 16,
        patience: int = 10,
        model_name: str = 'helmet_detector',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a helmet detection model.
        
        Args:
            data_yaml_path: Path to dataset YAML configuration
            epochs: Maximum number of training epochs
            image_size: Input image size for training
            batch_size: Training batch size
            patience: Early stopping patience
            model_name: Name for this training run
            **kwargs: Additional YOLO training arguments
            
        Returns:
            Dictionary with training results and paths
        """
        print("\n" + "="*60)
        print("TRAINING HELMET DETECTION MODEL")
        print("="*60)
        
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")
        
        model = YOLO(self.base_model)
        
        print(f"\nTraining Configuration:")
        print(f"   Base model: {self.base_model}")
        print(f"   Dataset: {data_yaml_path}")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {image_size}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {self.device}")
        print(f"   Patience: {patience}")
        print()
        
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            project=os.getenv('RUNS_PATH'),
            name=model_name,
            patience=patience,
            device=self.device,
            save=True,
            plots=True,
            **kwargs
        )
        
        # Get paths to saved models
        run_dir = Path(self.output_dir) / model_name
        best_model_path = run_dir / 'weights' / 'best.pt'
        last_model_path = run_dir / 'weights' / 'last.pt'
        
        print(f"\nTraining completed!")
        print(f"   Best model: {best_model_path}")
        print(f"   Last model: {last_model_path}")
        
        return {
            'results': results,
            'best_model_path': str(best_model_path),
            'last_model_path': str(last_model_path),
            'run_directory': str(run_dir)
        }
    
    def train_plate_model(
        self,
        data_yaml_path: str,
        epochs: int = 50,
        image_size: int = 640,
        batch_size: int = 16,
        patience: int = 10,
        model_name: str = 'plate_detector',
        augment: bool = True,
        augment_level: str = 'medium',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a license plate detection model.

        Args:
            data_yaml_path: Path to dataset YAML configuration
            epochs: Maximum number of training epochs
            image_size: Input image size for training
            batch_size: Training batch size
            patience: Early stopping patience
            model_name: Name for this training run
            augment: Enable data augmentation (default: True)
            augment_level: Augmentation intensity - 'light', 'medium', or 'heavy'
            **kwargs: Additional YOLO training arguments

        Returns:
            Dictionary with training results and paths
        """
        print("\n" + "="*60)
        print("TRAINING LICENSE PLATE DETECTION MODEL")
        print("="*60)

        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")

        model = YOLO(self.base_model)

        # Define augmentation presets optimized for plate detection
        # Note: We avoid horizontal flip (fliplr) as plates should not be mirrored
        aug_presets = {
            'light': {
                'hsv_h': 0.01,      # Slight hue variation
                'hsv_s': 0.5,       # Saturation variation
                'hsv_v': 0.3,       # Brightness variation
                'degrees': 5.0,     # Small rotation
                'translate': 0.1,   # Small translation
                'scale': 0.3,       # Scale variation
                'shear': 2.0,       # Small shear
                'perspective': 0.0001,
                'flipud': 0.0,      # No vertical flip
                'fliplr': 0.0,      # No horizontal flip (plates shouldn't be mirrored)
                'mosaic': 0.5,      # Moderate mosaic
                'mixup': 0.0,       # No mixup
            },
            'medium': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 10.0,
                'translate': 0.15,
                'scale': 0.5,
                'shear': 5.0,
                'perspective': 0.0005,
                'flipud': 0.0,
                'fliplr': 0.0,      # No horizontal flip
                'mosaic': 1.0,      # Full mosaic
                'mixup': 0.1,
                'copy_paste': 0.1,
            },
            'heavy': {
                'hsv_h': 0.02,
                'hsv_s': 0.9,
                'hsv_v': 0.5,
                'degrees': 15.0,
                'translate': 0.2,
                'scale': 0.7,
                'shear': 10.0,
                'perspective': 0.001,
                'flipud': 0.0,
                'fliplr': 0.0,      # No horizontal flip
                'mosaic': 1.0,
                'mixup': 0.2,
                'copy_paste': 0.2,
                'erasing': 0.2,     # Random erasing for robustness
            }
        }

        # Get augmentation config
        aug_config = {}
        if augment and augment_level in aug_presets:
            aug_config = aug_presets[augment_level]
            print(f"\nAugmentation: ENABLED ({augment_level} preset)")
        elif not augment:
            # Disable all augmentation
            aug_config = {
                'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0,
                'degrees': 0.0, 'translate': 0.0, 'scale': 0.0,
                'shear': 0.0, 'perspective': 0.0,
                'flipud': 0.0, 'fliplr': 0.0,
                'mosaic': 0.0, 'mixup': 0.0
            }
            print(f"\nAugmentation: DISABLED")

        print(f"\nTraining Configuration:")
        print(f"   Base model: {self.base_model}")
        print(f"   Dataset: {data_yaml_path}")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {image_size}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {self.device}")
        print(f"   Patience: {patience}")
        if augment:
            print(f"   Augmentation level: {augment_level}")
        print()

        # Merge augmentation config with any user-provided kwargs
        train_config = {**aug_config, **kwargs}

        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            project=os.getenv('RUNS_PATH'),
            name=model_name,
            patience=patience,
            device=self.device,
            save=True,
            plots=True,
            **train_config
        )
        
        run_dir = Path(self.output_dir) / model_name
        best_model_path = run_dir / 'weights' / 'best.pt'
        last_model_path = run_dir / 'weights' / 'last.pt'
        
        print(f"\nTraining completed!")
        print(f"   Best model: {best_model_path}")
        print(f"   Last model: {last_model_path}")
        
        return {
            'results': results,
            'best_model_path': str(best_model_path),
            'last_model_path': str(last_model_path),
            'run_directory': str(run_dir)
        }
    
    def train_both_models(
        self,
        helmet_data_yaml: str,
        plate_data_yaml: str,
        epochs: int = 50,
        batch_size: int = 16,
        plate_augment: bool = True,
        plate_augment_level: str = 'medium'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train both helmet and plate models sequentially.

        Args:
            helmet_data_yaml: Path to helmet dataset YAML
            plate_data_yaml: Path to plate dataset YAML
            epochs: Number of epochs for both models
            batch_size: Batch size for both models
            plate_augment: Enable augmentation for plate model
            plate_augment_level: Augmentation level for plate model

        Returns:
            Dictionary with results for both models
        """
        helmet_results = self.train_helmet_model(
            helmet_data_yaml,
            epochs=epochs,
            batch_size=batch_size
        )

        plate_results = self.train_plate_model(
            plate_data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            augment=plate_augment,
            augment_level=plate_augment_level
        )

        return {
            'helmet': helmet_results,
            'plate': plate_results
        }

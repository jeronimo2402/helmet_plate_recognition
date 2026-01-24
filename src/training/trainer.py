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
        batch_size: int = 16
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train both helmet and plate models sequentially.
        
        Args:
            helmet_data_yaml: Path to helmet dataset YAML
            plate_data_yaml: Path to plate dataset YAML
            epochs: Number of epochs for both models
            batch_size: Batch size for both models
            
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
            batch_size=batch_size
        )
        
        return {
            'helmet': helmet_results,
            'plate': plate_results
        }

"""
Dataset downloader module for Roboflow datasets.
Handles downloading and preparing datasets for training.
"""

import os
from typing import Optional
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()


class DatasetDownloader:
    """Downloads and manages datasets from Roboflow."""
    
    def __init__(self, api_key: str):
        """
        Initialize the dataset downloader.
        
        Args:
            api_key: Roboflow API key for authentication
        """
        self.api_key = api_key
        self.roboflow_client = Roboflow(api_key=api_key)
    
    def download_helmet_dataset(
        self, 
        workspace: str = "helmet-wparg",
        project: str = "helmet-detection-0xjjk-qsh2v",
        version: int = 1,
        output_format: str = "yolov8",
        save_path: Optional[str] = os.getenv('DATA_PATH') + "/Helmet-detection-1"
    ) -> str:
        """
        Download helmet detection dataset from Roboflow.
        
        Args:
            workspace: Roboflow workspace name
            project: Project name
            version: Dataset version number
            output_format: Format for labels (yolov8, coco, etc.)
            save_path: Optional custom save location
            
        Returns:
            Path to downloaded dataset
        """
        print(f"\nDownloading helmet dataset...")
        print(f"   Workspace: {workspace}")
        print(f"   Project: {project}")
        print(f"   Version: {version}")
        
        try:
            rf_project = self.roboflow_client.workspace(workspace).project(project)
            dataset_version = rf_project.version(version)
            
            if save_path:
                dataset = dataset_version.download(output_format, location=save_path)
            else:
                dataset = dataset_version.download(output_format)
            
            print(f"✓ Helmet dataset downloaded to: {dataset.location}")
            return dataset.location
            
        except Exception as e:
            raise RuntimeError(f"Failed to download helmet dataset: {str(e)}")
    
    def download_plate_dataset(
        self,
        workspace: str = "workspace-ikb1n",
        project: str = "bike-number-plate-3yvst-lllcl",
        version: int = 1,
        output_format: str = "yolov8",
        save_path: Optional[str] = os.getenv('DATA_PATH') + "/Bike-number-plate-1"
    ) -> str:
        """
        Download license plate detection dataset from Roboflow.
        
        Args:
            workspace: Roboflow workspace name
            project: Project name
            version: Dataset version number
            output_format: Format for labels
            save_path: Optional custom save location
            
        Returns:
            Path to downloaded dataset
        """
        print(f"\nDownloading plate dataset...")
        print(f"   Workspace: {workspace}")
        print(f"   Project: {project}")
        print(f"   Version: {version}")
        
        try:
            rf_project = self.roboflow_client.workspace(workspace).project(project)
            dataset_version = rf_project.version(version)
            
            if save_path:
                dataset = dataset_version.download(output_format, location=save_path)
            else:
                dataset = dataset_version.download(output_format)
            
            print(f"✓ Plate dataset downloaded to: {dataset.location}")
            return dataset.location
            
        except Exception as e:
            raise RuntimeError(f"Failed to download plate dataset: {str(e)}")
    
    def download_both_datasets(
        self,
        helmet_save_path: Optional[str] = os.getenv('DATA_PATH') + "/Helmet-detection-1",
        plate_save_path: Optional[str] = os.getenv('DATA_PATH') + "/Bike-number-plate-1"
    ) -> tuple[str, str]:
        """
        Download both helmet and plate datasets.
        
        Args:
            helmet_save_path: Optional path for helmet dataset
            plate_save_path: Optional path for plate dataset
            
        Returns:
            Tuple of (helmet_dataset_path, plate_dataset_path)
        """
        helmet_path = self.download_helmet_dataset(save_path=helmet_save_path)
        plate_path = self.download_plate_dataset(save_path=plate_save_path)
        
        return helmet_path, plate_path

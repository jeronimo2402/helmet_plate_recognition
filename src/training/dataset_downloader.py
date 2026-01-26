"""
Dataset downloader module for Roboflow datasets.
Handles downloading, merging, and preparing datasets for training.
"""

import os
import shutil
import yaml
from typing import Optional, List
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
        save_path: Optional[str] = None
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
        # Set default save path if not provided
        if save_path is None:
            data_path = os.getenv('DATA_PATH')
            if data_path:
                save_path = os.path.join(data_path, "Helmet-detection-1")
        
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
        save_path: Optional[str] = None
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
        # Set default save path if not provided
        if save_path is None:
            data_path = os.getenv('DATA_PATH')
            if data_path:
                save_path = os.path.join(data_path, "Bike-number-plate-1")
        
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
        helmet_save_path: Optional[str] = None,
        plate_save_path: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Download both helmet and plate datasets.
        
        Args:
            helmet_save_path: Optional path for helmet dataset
            plate_save_path: Optional path for plate dataset
            
        Returns:
            Tuple of (helmet_dataset_path, plate_dataset_path)
        """
        # Set default paths if not provided
        if helmet_save_path is None:
            data_path = os.getenv('DATA_PATH')
            if data_path:
                helmet_save_path = os.path.join(data_path, "Helmet-detection-1")
        
        if plate_save_path is None:
            data_path = os.getenv('DATA_PATH')
            if data_path:
                plate_save_path = os.path.join(data_path, "Bike-number-plate-1")
        
        helmet_path = self.download_helmet_dataset(save_path=helmet_save_path)
        plate_path = self.download_plate_dataset(save_path=plate_save_path)
        
        return helmet_path, plate_path

    def download_extended_plate_dataset(
        self,
        workspace: str = "workspace-ikb1n",
        project: str = "number-plate-xtpue-elloc",
        version: int = 1,
        output_format: str = "yolov8",
        save_path: Optional[str] = None
    ) -> str:
        """
        Download extended license plate dataset (larger, more varied distances).

        Args:
            workspace: Roboflow workspace name
            project: Project name
            version: Dataset version number
            output_format: Format for labels
            save_path: Optional custom save location

        Returns:
            Path to downloaded dataset
        """
        if save_path is None:
            data_path = os.getenv('DATA_PATH', 'data')
            save_path = os.path.join(data_path, "Number-plate-extended")

        print(f"\nDownloading extended plate dataset...")
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

            print(f"✓ Extended plate dataset downloaded to: {dataset.location}")
            return dataset.location

        except Exception as e:
            raise RuntimeError(f"Failed to download extended plate dataset: {str(e)}")

    def merge_plate_datasets(
        self,
        dataset_paths: List[str],
        output_path: Optional[str] = None,
        class_name: str = "license_plate"
    ) -> str:
        """
        Merge multiple plate datasets into a single unified dataset.

        This combines images and labels from multiple sources, normalizing
        class names to ensure consistency.

        Args:
            dataset_paths: List of paths to datasets to merge
            output_path: Where to save merged dataset
            class_name: Unified class name for plates (default: license_plate)

        Returns:
            Path to merged dataset
        """
        if output_path is None:
            data_path = os.getenv('DATA_PATH', 'data')
            output_path = os.path.join(data_path, "Plate-merged")

        print(f"\nMerging {len(dataset_paths)} plate datasets...")

        # Create output directory structure
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)

        total_images = {'train': 0, 'valid': 0, 'test': 0}

        for dataset_idx, dataset_path in enumerate(dataset_paths):
            print(f"   Processing dataset {dataset_idx + 1}: {dataset_path}")

            for split in ['train', 'valid', 'test']:
                # Check both common naming conventions
                split_dir = split
                if not os.path.exists(os.path.join(dataset_path, split, 'images')):
                    # Try 'val' instead of 'valid'
                    if split == 'valid' and os.path.exists(os.path.join(dataset_path, 'val', 'images')):
                        split_dir = 'val'
                    else:
                        continue

                images_src = os.path.join(dataset_path, split_dir, 'images')
                labels_src = os.path.join(dataset_path, split_dir, 'labels')

                if not os.path.exists(images_src):
                    continue

                images_dst = os.path.join(output_path, split, 'images')
                labels_dst = os.path.join(output_path, split, 'labels')

                # Copy images and labels with prefixed names to avoid collisions
                for img_file in os.listdir(images_src):
                    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        continue

                    # Create unique filename with dataset prefix
                    new_name = f"ds{dataset_idx}_{img_file}"

                    # Copy image
                    shutil.copy2(
                        os.path.join(images_src, img_file),
                        os.path.join(images_dst, new_name)
                    )

                    # Copy and normalize label (change class ID to 0)
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_src_path = os.path.join(labels_src, label_file)

                    if os.path.exists(label_src_path):
                        new_label_name = f"ds{dataset_idx}_{label_file}"
                        self._normalize_label_file(
                            label_src_path,
                            os.path.join(labels_dst, new_label_name)
                        )

                    total_images[split] += 1

        # Create data.yaml for merged dataset
        yaml_content = {
            'path': os.path.abspath(output_path),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': 1,
            'names': [class_name]
        }

        yaml_path = os.path.join(output_path, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"✓ Merged dataset created at: {output_path}")
        print(f"   Train images: {total_images['train']}")
        print(f"   Valid images: {total_images['valid']}")
        print(f"   Test images: {total_images['test']}")
        print(f"   Total: {sum(total_images.values())}")

        return output_path

    def _normalize_label_file(self, src_path: str, dst_path: str) -> None:
        """
        Copy a YOLO label file, normalizing all class IDs to 0.

        Since we're merging plate datasets that might have different class IDs,
        we normalize everything to class 0 (license_plate).
        """
        with open(src_path, 'r') as f:
            lines = f.readlines()

        normalized_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Replace class ID with 0, keep bbox coordinates
                parts[0] = '0'
                normalized_lines.append(' '.join(parts) + '\n')

        with open(dst_path, 'w') as f:
            f.writelines(normalized_lines)

    def download_and_merge_plate_datasets(self) -> str:
        """
        Download both plate datasets and merge them into one.

        This is the recommended method for training a robust plate detector.

        Returns:
            Path to merged plate dataset
        """
        print("\n" + "="*50)
        print("Downloading and merging plate datasets")
        print("="*50)

        # Download both datasets
        bike_plate_path = self.download_plate_dataset()
        extended_plate_path = self.download_extended_plate_dataset()

        # Merge them
        merged_path = self.merge_plate_datasets([bike_plate_path, extended_plate_path])

        return merged_path
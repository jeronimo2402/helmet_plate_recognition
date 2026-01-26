"""
Training CLI for helmet and license plate detection models.
Usage:
    python train.py --mode helmet --epochs 50
    python train.py --mode plate --epochs 50
    python train.py --mode both --epochs 50
"""

import argparse
import os
from src.training import DatasetDownloader, ModelTrainer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train helmet and license plate detection models'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['helmet', 'plate', 'both'],
        required=True,
        help='Which model to train: helmet, plate, or both'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('ROBOFLOW_API_KEY'),
        help='Roboflow API key (can also set ROBOFLOW_API_KEY env var)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size (default: 16)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for training (default: cpu)'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip dataset download (use existing datasets)'
    )
    
    parser.add_argument(
        '--helmet-dataset',
        type=str,
        help='Path to existing helmet dataset (if skip-download)'
    )
    
    parser.add_argument(
        '--plate-dataset',
        type=str,
        help='Path to existing plate dataset (if skip-download)'
    )

    parser.add_argument(
        '--merge-plate-datasets',
        action='store_true',
        default=True,
        help='Download and merge multiple plate datasets for better detection (default: enabled)'
    )

    parser.add_argument(
        '--no-merge-plate-datasets',
        action='store_false',
        dest='merge_plate_datasets',
        help='Use only the original bike plate dataset (not recommended)'
    )

    parser.add_argument(
        '--augment',
        action='store_true',
        default=True,
        help='Enable data augmentation for plate model (default: enabled)'
    )

    parser.add_argument(
        '--no-augment',
        action='store_false',
        dest='augment',
        help='Disable data augmentation for plate model'
    )

    parser.add_argument(
        '--augment-level',
        type=str,
        choices=['light', 'medium', 'heavy'],
        default='medium',
        help='Augmentation intensity: light, medium, or heavy (default: medium)'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("  HELMET & LICENSE PLATE DETECTION - MODEL TRAINING")
    print("="*70)
    
    # Initialize trainer
    trainer = ModelTrainer(
        base_model='yolov8n.pt',
        device=args.device,
        output_dir=os.getenv('RUNS_PATH')
    )
    
    # Download datasets if needed
    helmet_data_path = None
    plate_data_path = None
    
    if not args.skip_download:
        if not args.api_key:
            raise ValueError(
                "Roboflow API key required. Set --api-key or ROBOFLOW_API_KEY env variable"
            )
        
        print("\nDownloading datasets from Roboflow...")
        downloader = DatasetDownloader(args.api_key)
        
        if args.mode in ['helmet', 'both']:
            helmet_data_path = downloader.download_helmet_dataset()
            helmet_yaml = os.path.join(helmet_data_path, 'data.yaml')
        
        if args.mode in ['plate', 'both']:
            if args.merge_plate_datasets:
                # Download and merge both plate datasets for better detection
                plate_data_path = downloader.download_and_merge_plate_datasets()
            else:
                # Use only the original bike plate dataset
                plate_data_path = downloader.download_plate_dataset()
            plate_yaml = os.path.join(plate_data_path, 'data.yaml')
    else:
        # Use provided dataset paths
        if args.mode in ['helmet', 'both']:
            if not args.helmet_dataset:
                raise ValueError("--helmet-dataset required when using --skip-download")
            helmet_yaml = os.path.join(args.helmet_dataset, 'data.yaml')
        
        if args.mode in ['plate', 'both']:
            if not args.plate_dataset:
                raise ValueError("--plate-dataset required when using --skip-download")
            plate_yaml = os.path.join(args.plate_dataset, 'data.yaml')
    
    # Train models
    results = {}
    
    if args.mode == 'helmet':
        print("\n" + "="*70)
        print("Training HELMET detection model only")
        print("="*70)
        results['helmet'] = trainer.train_helmet_model(
            helmet_yaml,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    elif args.mode == 'plate':
        print("\n" + "="*70)
        print("Training LICENSE PLATE detection model only")
        print("="*70)
        results['plate'] = trainer.train_plate_model(
            plate_yaml,
            epochs=args.epochs,
            batch_size=args.batch_size,
            augment=args.augment,
            augment_level=args.augment_level
        )
    
    elif args.mode == 'both':
        print("\n" + "="*70)
        print("Training BOTH models sequentially")
        print("="*70)
        results = trainer.train_both_models(
            helmet_yaml,
            plate_yaml,
            epochs=args.epochs,
            batch_size=args.batch_size,
            plate_augment=args.augment,
            plate_augment_level=args.augment_level
        )
    
    # Print final summary
    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    
    if 'helmet' in results:
        print(f"\n Helmet Model:")
        print(f"   Best: {results['helmet']['best_model_path']}")
        print(f"   Last: {results['helmet']['last_model_path']}")
    
    if 'plate' in results:
        print(f"\n Plate Model:")
        print(f"   Best: {results['plate']['best_model_path']}")
        print(f"   Last: {results['plate']['last_model_path']}")
    
    print("\n Next steps:")
    print("   1. Copy trained models to models/ directory:")
    if 'helmet' in results:
        print(f"      cp {results['helmet']['best_model_path']} models/helmet_model.pt")
    if 'plate' in results:
        print(f"      cp {results['plate']['best_model_path']} models/plate_model.pt")
    print("   2. Run predictions: python predict.py --image <path>")
    print()


if __name__ == '__main__':
    main()

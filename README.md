# Helmet & License Plate Detection System

A complete AI-powered system for detecting helmet violations and reading license plates from motorcycle riders, with intelligent spatial matching to correctly pair each person with their vehicle.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-Poetry-blue)](https://python-poetry.org/)
[![YOLOv8](https://img.shields.io/badge/model-YOLOv8-green)](https://github.com/ultralytics/ultralytics)

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Training Models](#training-models)
- [Training Analysis](#training-analysis)
- [Web Interface](#web-interface)
- [Testing](#testing)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)

---

## Features

### Core Capabilities

- **Helmet Detection**: YOLOv8-based detection of people with/without helmets
- **License Plate Detection**: Accurate plate localization using trained YOLO model
- **Intelligent Spatial Matching**: Correctly pairs each person with THEIR specific plate (not just any plate in the image)
- **OCR Text Reading**: EasyOCR with preprocessing for plate text extraction
- **Report Generation**: CSV, JSON, and Excel report formats with annotated images

### Training & Analysis

- **Complete Training Pipeline**: Train both helmet and plate models from scratch
- **Dataset Management**: Automated download from Roboflow
- **Training Metrics Visualization**: Custom plots for losses, metrics, learning rate, and efficiency
- **GPU Support**: Full CUDA acceleration for training and inference
- **Augmentation Control**: Configurable data augmentation levels

### Deployment

- **Production-Ready Inference**: Optimized prediction pipeline with CLI
- **Batch Processing**: Handle single images or entire folders
- **Flask Web Interface**: User-friendly web UI for image upload and processing
- **Flexible Output**: Multiple report formats and annotated images

---

## Project Structure

```text
helmet_plate_recognition/
├── src/
│   ├── __init__.py                    # Main package exports
│   ├── core/                          # Core detection logic
│   │   ├── __init__.py
│   │   ├── helmet_detector.py         # Helmet detection (YOLOv8)
│   │   ├── plate_detector.py          # License plate detection (YOLOv8)
│   │   ├── plate_reader.py            # OCR for plates (EasyOCR)
│   │   └── spatial_matching.py        # Person-to-plate matching algorithm
│   ├── training/                      # Training modules
│   │   ├── __init__.py
│   │   ├── dataset_downloader.py      # Roboflow dataset downloader
│   │   └── trainer.py                 # Model training logic
│   ├── utils/                         # Utilities
│   │   ├── __init__.py
│   │   ├── image_annotator.py         # Image annotation with bounding boxes
│   │   ├── report_generator.py        # Report creation (CSV/JSON/Excel)
│   │   └── training_analyzer.py       # Training metrics visualization
│   └── tests/                         # Test scripts
│       ├── test_plate_only.py         # Isolated plate detection test
│       └── test_pipeline.py           # Full pipeline debug script
├── models/                            # Trained model weights
│   ├── helmet_model.pt                # Trained helmet detector
│   ├── plate_model.pt                 # Trained plate detector
|   └──runs/detect/                    # Training outputs
|      ├──helmet_detector/             # Helmet model training results
|      │   ├── weights/                # Model checkpoints
|      │   ├── results.csv             # Training metrics
|      │   └── analysis/               # Custom training plots
|      └──plate_detector/              # Plate model training results
|          ├── weights/                # Model checkpoints
|          ├── results.csv             # Training metrics
|          └── analysis/               # Custom training plots
├── data/                              # Input images and datasets
│   └── test_data/                     # Test images
├── outputs/                           # Generated outputs
│   ├── reports/                       # CSV/JSON/Excel reports
│   └── web_reports/                   # CSV/JSON/Excel reports
├── templates/                         # Flask HTML templates
│   └── index.html                     # Web interface
├── notebooks/                         # Jupyter notebooks
│   └── helmet_detection.ipynb         # Original exploration notebook
├── train.py                           # Training CLI script
├── predict.py                         # Prediction CLI script
├── app.py                             # Flask web application
├── pyproject.toml                     # Poetry configuration
├── poetry.lock                        # Locked dependencies
└── .env                               # Environment variables
```

---

## Installation

### Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA 12.1+ (optional, for GPU acceleration)
- Poetry (Python dependency manager)

### Step 1: Install Poetry

```bash
# Linux/macOS/WSL
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Step 2: Clone and Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd helmet_plate_recognition

# Install all dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Step 3: Environment Configuration

Create a `.env` file in the project root:

```bash
# Roboflow API key (required for training)
ROBOFLOW_API_KEY="your_roboflow_api_key_here"

# Optional: Custom paths
DATA_PATH="./data"
RUNS_PATH="./models/runs"
```

**Get your Roboflow API key:**

1. Go to [Roboflow](https://roboflow.com/)
2. Sign up/login
3. Navigate to Settings → API
4. Copy your API key

### Step 4: Verify Installation

```bash
python -c "
from src.core import HelmetDetector, PlateDetector, PlateReader, SpatialMatcher
from src.utils import ReportGenerator, ImageAnnotator, TrainingAnalyzer
from src.training import DatasetDownloader, ModelTrainer
print('✓ All modules imported successfully!')
"
```

---

## Quick Start

### Option 1: Use Pre-trained Models (Recommended)

If you have pre-trained models, place them in the `models/` directory:

```bash
models/
├── helmet_model.pt
└── plate_model.pt
```

Then run prediction:

```bash
poetry run python predict.py --image data/test_data/test_picture_4.jpeg --device cuda
```

### Option 2: Train from Scratch

```bash
# Train both models (takes ~3-5 minutes on GPU)
poetry run python train.py --model both --epochs 50 --device cuda

# Copy trained models to production directory
cp models/runs/detect/helmet_detector/weights/best.pt models/helmet_model.pt
cp models/runs/detect/plate_detector/weights/best.pt models/plate_model.pt

# Run prediction
poetry run python predict.py --image data/test.jpg --device cuda
```

---

## Usage

### Command Line Interface

#### Single Image Processing

```bash
# Basic usage (CPU)
poetry run python predict.py --image data/test.jpg

# With GPU acceleration
poetry run python predict.py --image data/test.jpg --device cuda

# Custom confidence threshold
poetry run python predict.py --image data/test.jpg --confidence 0.3 --device cuda

# Specify output format
poetry run python predict.py --image data/test.jpg --output-format json --device cuda
```

#### Batch Processing (Folder)

```bash
# Process all images in a folder
poetry run python predict.py --folder data/batch/ --device cuda

# With custom models
poetry run python predict.py \
  --folder data/batch/ \
  --helmet-model models/custom_helmet.pt \
  --plate-model models/custom_plate.pt \
  --device cuda
```

#### Output Options

The system generates:

- **Reports**: CSV/JSON/Excel files in `outputs/reports/` and `outputs/web_reports/`
- **Annotated Images**: Images with bounding boxes in `outputs/reports/images/` and `outputs/web_reports/images/`

**Report columns:**

| Column                        | Description                              |
|-------------------------------|------------------------------------------|
| `image_file`                  | Name of the image file                   |
| `person_id`                   | Person ID (1, 2, 3, etc.)                |
| `has_helmet`                  | True if person has helmet, False if not  |
| `helmet_status`               | "WITH_HELMET" or "WITHOUT_HELMET"        |
| `detection_confidence`        | AI confidence score (0.0 to 1.0)         |
| `license_plate`               | Plate number (or "NO_PLATE_DETECTED")    |
| `plate_confidence`            | AI confidence score (0.0 to 1.0)         |
| `plate_matched`               | True if plate matched, False if not      |
| `person_box`                  | [x1, y1, x2, y2] coordinates             |
| `matched_plate_box`           | [x1, y1, x2, y2] coordinates             |
| `annotated_image`             | Path to annotated image                  |

---

## Training Models

### Basic Training

```bash
# Train both models with default settings
poetry run python train.py --model both --epochs 50 --device cuda

# Train only helmet detector
poetry run python train.py --model helmet --epochs 50 --device cuda

# Train only plate detector
poetry run python train.py --model plate --epochs 50 --device cuda
```

### Advanced Training Options

```bash
# Custom batch size and image size
poetry run python train.py \
  --model both \
  --epochs 100 \
  --batch-size 32 \
  --image-size 640 \
  --device cuda

# Heavy data augmentation for better generalization
poetry run python train.py \
  --model plate \
  --augment-level heavy \
  --epochs 100 \
  --device cuda

# No augmentation (not recommended)
poetry run python train.py \
  --model plate \
  --no-augment \
  --epochs 50 \
  --device cuda

# Custom output directory
poetry run python train.py \
  --model both \
  --runs-dir custom_runs/ \
  --device cuda
```

### Training Process

The training script will:

1. **Download datasets** from Roboflow (if not already present)
2. **Train models** with progress bars and metrics
3. **Save checkpoints** to `models/runs/detect/`
4. **Generate analysis plots** automatically after training
5. **Display summary** with best model paths

### After Training

```bash
# Copy best models to production
cp models/runs/detect/helmet_detector/weights/best.pt models/helmet_model.pt
cp models/runs/detect/plate_detector/weights/best.pt models/plate_model.pt
```

---

## Training Analysis

The system automatically generates custom training analysis plots after training completes. You can also generate them manually:

```bash
# View analysis results
ls models/runs/detect/helmet_detector/analysis/
ls models/runs/detect/plate_detector/analysis/
```

### Generated Plots

1. **`loss_curves.png`**: Training and validation losses (box, class, DFL)
2. **`metrics_evolution.png`**: Precision, recall, mAP@50, mAP@50-95 over epochs
3. **`learning_rate.png`**: Learning rate schedule visualization
4. **`overfitting_analysis.png`**: Train-validation gap analysis
5. **`training_time.png`**: Time per epoch and cumulative training time
6. **`efficiency_metrics.png`**: Metrics improvement vs training time

### Training Summary

The script displays:

- Total training time (seconds and minutes)
- Average time per epoch
- Final metrics (precision, recall, mAP)
- Best epoch and its metrics

---

## Web Interface

Launch the Flask web application for a user-friendly interface:

```bash
# Start web server (CPU)
poetry run python app.py

# Start with GPU support
poetry run python app.py --device cuda

# Custom port
poetry run python app.py --port 8080 --device cuda
```

Access the interface at: `http://localhost:5000`

### Web Interface Features

- **Image Upload**: Drag and drop or browse for images
- **Real-time Processing**: View results immediately
- **Annotated Images**: See bounding boxes and labels
- **Download Reports**: Get CSV/JSON reports
- **Violation Summary**: Quick overview of helmet violations

---

## Testing

### Test Plate Detection in Isolation

```bash
poetry run python src/tests/test_plate_only.py \
  --image data/test_data/test_picture_4.jpeg \
  --conf 0.1
```

This script:

- Tests plate detection model independently
- Shows all detections with confidence scores
- Saves annotated results to `src/tests/results/`
- Helps debug plate detection issues

### Test Full Pipeline

```bash
poetry run python src/tests/test_pipeline.py \
  --image data/test_data/test_picture_4.jpeg
```

This script:

- Traces the complete detection pipeline
- Shows helmet detection results
- Shows cropped regions sent to plate detector
- Compares crop-based vs full-image detection
- Saves debug crops for inspection

---

## Configuration

### Model Paths

Default model paths can be customized:

```bash
poetry run python predict.py \
  --helmet-model path/to/custom_helmet.pt \
  --plate-model path/to/custom_plate.pt \
  --image data/test.jpg
```

### Confidence Thresholds

Adjust detection sensitivity:

```bash
# More lenient (detects more, but may have false positives)
poetry run python predict.py --image data/test.jpg --confidence 0.2

# More strict (fewer detections, higher accuracy)
poetry run python predict.py --image data/test.jpg --confidence 0.5
```

### Device Selection

```bash
# Use CPU
poetry run python predict.py --image data/test.jpg --device cpu

# Use GPU (CUDA)
poetry run python predict.py --image data/test.jpg --device cuda

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Troubleshooting

### Installation Issues

#### "No module named 'src'"

```bash
# Make sure you're in the project root and dependencies are installed
poetry install
poetry shell
```

#### "CUDA not available" but you have NVIDIA GPU

```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Reinstall PyTorch with CUDA support
poetry add torch torchvision --source pytorch-cuda
```

#### "Could not load image"

Check that:

- Image file exists and path is correct
- Image format is supported (.jpg, .png, .jpeg, .bmp)
- File permissions allow reading

```bash
# Verify image
file data/test.jpg
ls -lh data/test.jpg
```

### Training Issues

#### "Roboflow API key not found"

```bash
# Make sure .env file exists and contains your key
cat .env | grep ROBOFLOW_API_KEY

# Or set it directly
export ROBOFLOW_API_KEY="your_key_here"
```

#### "Out of memory" during training

```bash
# Reduce batch size
poetry run python train.py --model both --batch-size 8 --device cuda

# Or use CPU (slower but no memory limit)
poetry run python train.py --model both --device cpu
```

#### Training is very slow

- **CPU training**: Expected, takes 30-60 minutes per model
- **GPU training**: Should take 2-5 minutes per model
- Check GPU utilization: `nvidia-smi`

### Prediction Issues

#### "Model file not found"

```bash
# Verify models exist
ls -lh models/

# Should show:
# helmet_model.pt
# plate_model.pt

# If missing, train models or copy from training output
cp models/runs/detect/helmet_detector/weights/best.pt models/helmet_model.pt
cp models/runs/detect/plate_detector/weights/best.pt models/plate_model.pt
```

#### No plates detected

- Try lower confidence threshold: `--confidence 0.1`
- Check image quality (plates should be visible and readable)
- Use test scripts to debug: `python src/tests/test_plate_only.py`
- Verify plate model is trained on similar data

#### EasyOCR is slow on first run

EasyOCR downloads language models (~100MB) on first use. This is normal and only happens once.

### Web Interface Issues

#### "Address already in use"

```bash
# Kill existing Flask process
pkill -f app.py

# Or use different port
poetry run python app.py --port 8080
```

#### Images not displaying

- Check `outputs/` directories exist
- Verify file permissions
- Check browser console for errors

---

## Performance

### Training Time

| Hardware       | Helmet Model | Plate Model | Total   |
|----------------|--------------|-------------|---------|
| CPU            | ~30-40 min   | ~25-35 min  | ~1 hour |
| GPU (RTX 5070) | ~1.3 min     | ~2.7 min    | ~4 min  |

### Inference Time

| Hardware | Time per Image |
|----------|----------------|
| CPU      | ~100-200ms     |
| GPU      | ~20-50ms       |

> **Note:** Actual performance depends on dataset quality and training parameters

---

## Development

### Project Dependencies

Main libraries:

- **ultralytics**: YOLOv8 implementation
- **easyocr**: OCR for license plates
- **opencv-python**: Image processing
- **pandas**: Data handling
- **matplotlib/seaborn**: Visualization
- **flask**: Web interface
- **python-dotenv**: Environment variables

### Adding New Features

```bash
# Add new dependency
poetry add library-name

# Update dependencies
poetry update

# Run in development mode
poetry run python predict.py --image data/test.jpg
```

### Code Structure

- **`src/core/`**: Core detection and matching logic
- **`src/training/`**: Model training and dataset management
- **`src/utils/`**: Utilities for reports, annotations, and analysis
- **`src/tests/`**: Test and debug scripts

---

## Support

For issues and questions:

- Check the [Troubleshooting](#troubleshooting) section
- Review test scripts in `src/tests/`
- Open an issue on GitHub
- Contact us at:
  - [jeromosque@outlook.com](mailto:jeromosque@outlook.com)
  - [alejoalvarezgil2@gmail.com](mailto:alejoalvarezgil2@gmail.com)
  - [jndggiraldo@gmail.com](mailto:jndggiraldo@gmail.com)
  
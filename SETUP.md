# Setup Guide - Helmet & License Plate Detection System

## Complete Setup from Scratch

### Step 1: Environment Setup

#### Install Dependencies with Poetry

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Step 2: Get Roboflow API Key

1. Go to [Roboflow](https://roboflow.com/)
2. Sign up/login
3. Navigate to Settings → API
4. Copy your API key

### Step 3: Train Models

#### Option A: Quick Training (CPU)

```bash
python train.py --mode both --epochs 50 --device cpu
```

This will:

- Download helmet dataset (351 images)
- Download plate dataset (620 images)
- Train helmet detector (~30-40 min on CPU)
- Train plate detector (~25-35 min on CPU)
- Save models to `runs/detect/`

#### Option B: Fast Training (GPU)

```bash
python train.py --mode both --epochs 50 --device cuda
```

Much faster (~5-10 min per model)

### Step 4: Deploy Trained Models

After training completes, the script will show you where models are saved. Copy them:

```bash
# Copy best-performing models to production directory
cp runs/detect/helmet_detector/weights/best.pt models/helmet_model.pt
cp runs/detect/plate_detector/weights/best.pt models/plate_model.pt
```

### Step 5: Test the System

#### Prepare Test Image

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Copy a test image (with motorcycles/helmets) to data/
cp /path/to/your/test_image.jpg data/test.jpg
```

#### Run Prediction

```bash
python predict.py --image data/test.jpg
```

You should see:

```text
Processing: test.jpg
   Found 2 people
   Found 1 license plates
   Matched 1 people to plates
   Person #1: VIOLATION (no helmet) - Plate: ABC123
   Person #2: Compliant (helmet) - Plate: XYZ789
   
Report saved: outputs/reports/report_20260121_025900.csv
```

### Step 6: Verify Installation

Run the verification script:

```bash
python -c "
from src.core import HelmetDetector, PlateDetector, PlateReader, SpatialMatcher
from src.utils import ReportGenerator
from src.training import DatasetDownloader, ModelTrainer
print('All modules imported successfully')
"
```

## Troubleshooting Setup

### Issue: "No module named 'ultralytics'"

```bash
poetry install
# or if already installed, update
poetry update
```

### Issue: "CUDA not available"

This is normal if you don't have NVIDIA GPU. System will use CPU automatically.
To verify:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Issue: "Model file not found"

Make sure you've copied trained models to `models/` directory:

```bash
ls models/
# Should show: helmet_model.pt  plate_model.pt
```

### Issue: EasyOCR downloads models slowly

First run of EasyOCR downloads recognition models (~100MB). This is normal and only happens once.

## Quick Reference

### Directory Structure After Setup

```text
helmet_plate_recognition/
├── models/
│   ├── helmet_model.pt    ← Trained helmet detector
│   └── plate_model.pt     ← Trained plate detector
├── data/
│   └── test.jpg           ← Your test images
├── outputs/
│   └── reports/           ← Generated CSV/JSON reports
├── runs/
│   └── detect/            ← Training outputs
│       ├── helmet_detector/
│       └── plate_detector/
├── pyproject.toml         ← Poetry configuration
└── poetry.lock            ← Locked dependencies
```

### Essential Commands

```bash
# Train models
python train.py --mode both --epochs 50

# Predict single image
python predict.py --image data/test.jpg

# Predict folder
python predict.py --folder data/batch/

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

1. **Test with your own images**: Place motorcycle images in `data/`
2. **Adjust confidence thresholds**: Use `--confidence 0.3` for more lenient detection
3. **Batch processing**: Process entire folders for production use
4. **Customize training**: Modify `train.py` for custom epochs/batch sizes

## Performance Expectations

### Training Time (CPU)

- Helmet model: ~30-40 minutes
- Plate model: ~25-35 minutes
- Total: ~1 hour

### Training Time (GPU)

- Helmet model: ~5-8 minutes
- Plate model: ~4-6 minutes
- Total: ~15 minutes

### Inference Time

- CPU: ~100-200ms per image
- GPU: ~20-50ms per image

### Model Performance

- Helmet Detection: ~67% mAP50
- Plate Detection: ~99% mAP50
- OCR Accuracy: Depends on image quality (high-res plates: >80%)

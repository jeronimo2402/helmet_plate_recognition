# 🏍️ Helmet & License Plate Detection System

A complete AI-powered system for detecting helmet violations and reading license plates from motorcycle riders, with intelligent spatial matching to correctly pair each person with their vehicle.

## ✨ Features

### Core Capabilities

- **Helmet Detection**: YOLOv8-based detection of people with/without helmets
- **License Plate Detection**: Accurate plate localization using trained YOLO model
- **Intelligent Spatial Matching**: Correctly pairs each person with THEIR specific plate (not just any plate in the image)
- **OCR Text Reading**: EasyOCR with preprocessing for plate text extraction
- **Report Generation**: CSV, JSON, and Excel report formats

### Training & Deployment

- **Complete Training Pipeline**: Train both helmet and plate models from scratch
- **Dataset Management**: Automated download from Roboflow
- **Production-Ready Inference**: Optimized prediction pipeline with CLI
- **Batch Processing**: Handle single images or entire folders

## 🏗️ Project Structure

```text
helmet_plate_recognition/
├── src/
│   ├── __init__.py              # Main package exports
│   ├── core/                    # Core detection logic
│   │   ├── __init__.py
│   │   ├── helmet_detector.py   # Helmet detection
│   │   ├── plate_detector.py    # License plate detection
│   │   ├── plate_reader.py      # OCR for plates
│   │   └── spatial_matching.py  # Person-to-plate matching
│   ├── training/                # Training modules
│   │   ├── __init__.py
│   │   ├── dataset_downloader.py
│   │   └── trainer.py
│   └── utils/                   # Utilities
│       ├── __init__.py
│       └── report_generator.py  # Report creation
├── models/                      # Trained model weights
│   ├── helmet_model.pt
│   └── plate_model.pt
├── data/                        # Input images
├── outputs/                     # Generated reports
├── notebooks/                   # Jupyter notebooks
├── train.py                     # Training CLI
├── predict.py                   # Prediction CLI
├── pyproject.toml               # Poetry configuration
└── poetry.lock                  # Locked dependencies
```

## � Installation

### Install Dependencies with Poetry

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## 📦 Required Models

Before running the program, you need:

1. **Trained helmet detection model** (`models/helmet_model.pt`)
   - Train this using your Roboflow dataset in the notebook
   - Copy the trained model to `models/` folder

2. **License plate detection model** (optional)
   - Can use pre-trained YOLO models
   - Or train your own on a plate dataset

## 🎯 How to Use

### Option 1: Process a Single Image

Edit `main.py` and set:

```python
SINGLE_IMAGE_MODE = True
image_path = 'data/test_image.jpg'
```

Then run:

```bash
poetry run python main.py
```

### Option 2: Process Multiple Images

Edit `main.py` and set:

```python
SINGLE_IMAGE_MODE = False
folder_path = 'data/'
```

Then run:

```bash
poetry run python main.py
```

### Choose Output Format

You can generate reports in different formats:

- `'csv'` - Excel-compatible spreadsheet
- `'json'` - JSON format for programming
- `'excel'` - Native Excel file

Change this in `main.py`:

```python
report_path = process_single_image(image_path, output_format='csv')
```

## 📊 Report Contents

The generated report includes:

| Column | Description |
|--------|-------------|

| `image_file` | Name of the image file |
| `has_helmet` | True if person has helmet, False if not |
| `license_plate` | Plate number (or "NO_PLATE_DETECTED") |
| `detection_confidence` | How confident the AI is (0.0 to 1.0) |

## 🔧 Training Your Models

### Train Helmet Detector

1. Open `notebooks/01_train_helmet_detector.ipynb`
2. Run all cells to train the model
3. Copy the trained model from `runs/detect/helmet_detector/weights/best.pt` to `models/helmet_model.pt`

### Train Plate Detector (Optional)

1. Get a license plate dataset (e.g., from Roboflow)
2. Create a training notebook similar to the helmet one
3. Save the trained model to `models/plate_model.pt`

## 📚 Understanding the Code

### Variable Names Explanation

All variable names are designed to be self-explanatory:

- `helmet_detection_system` - The object that detects helmets
- `plate_text_reader` - The object that reads text from plates
- `list_of_people_found` - A list containing all detected people
- `person_has_helmet` - Boolean (True/False) if person wears helmet
- `license_plate_text` - The text read from the license plate
- `bounding_box_coordinates` - Rectangle coordinates [left, top, right, bottom]
- `detection_confidence` - Number from 0.0 to 1.0 showing AI confidence

### How Each Module Works

**`helmet_detector.py`**: Uses YOLO AI model to find people and check if they have helmets

**`plate_detector.py`**: Uses YOLO to find license plates in the image

**`plate_reader.py`**: Uses EasyOCR to read text from cropped plate images

**`report_generator.py`**: Collects all information and saves it as a file

## 🛠️ Common Commands

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run the main program
poetry run python main.py

# Add a new library
poetry add library-name

# Update all libraries
poetry update

# Run Jupyter notebook
poetry run jupyter notebook

# Exit virtual environment
exit
```

## 🐛 Troubleshooting

### "No module named 'src'"

Make sure you're in the project root folder and run:

```bash
poetry install
```

### "Could not load image"

Check that:

- Image file exists in the `data/` folder
- Path in `main.py` is correct
- Image format is supported (.jpg, .png, .jpeg, .bmp)

### "Model not found"

Make sure you have:

- Trained the helmet detection model
- Copied it to `models/helmet_model.pt`
- Path is correct in the code

### EasyOCR is slow on first run

EasyOCR downloads language models on first use. This is normal and only happens once.

## 📝 Next Steps

1. ✅ Train your helmet detection model using the notebook
2. ✅ Copy the trained model to `models/` folder
3. ✅ Put test images in `data/` folder
4. ✅ Run `main.py` to process images
5. ✅ Check the generated report in `outputs/reports/`

## 🔧 Configuration

Configuration is handled through CLI parameters:

- `--helmet-model`, `--plate-model`: Model paths
- `--confidence`: Detection confidence threshold
- `--output-format`: Report format (csv/json/excel)
- `--device`: CPU or CUDA/

## 🤝 Contributing

Feel free to improve the code! Some ideas:

- Add video processing capability
- Improve plate reading accuracy
- Add visualization (draw boxes on images)
- Create a web interface
- Add support for multiple languages in plate reading

## 📄 License

This project is for educational purposes.

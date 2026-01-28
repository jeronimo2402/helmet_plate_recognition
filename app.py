"""
Web interface for helmet violation detection.
Run with: python app.py
Access at: http://localhost:5000
"""

import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify, url_for
from werkzeug.utils import secure_filename
from src.core import HelmetDetector, PlateDetector, PlateReader, SpatialMatcher, ImageProcessor
from src.utils import ReportGenerator, ImageAnnotator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'templates/uploads'
app.config['OUTPUT_FOLDER'] = 'outputs/web_reports'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['OUTPUT_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['OUTPUT_FOLDER'], 'reports'), exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

# Global model instances (loaded once)
helmet_detector = None
plate_detector = None
plate_reader = None
spatial_matcher = None
image_processor = None
image_annotator = None
report_generator = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load all models (called once at startup)."""
    global helmet_detector, plate_detector, plate_reader, spatial_matcher, image_processor, image_annotator, report_generator

    print("Loading models...")

    helmet_detector = HelmetDetector(
        'models/helmet_model.pt',
        device='cuda',
        confidence_threshold=0.25
    )

    plate_detector = PlateDetector(
        'models/plate_model.pt',
        device='cuda',
        confidence_threshold=0.25
    )

    plate_reader = PlateReader(
        supported_languages=['en'],
        gpu=True
    )

    spatial_matcher = SpatialMatcher(
        horizontal_threshold=200,
        vertical_overlap_threshold=50
    )

    image_processor = ImageProcessor(
        helmet_detector,
        plate_detector,
        plate_reader,
        spatial_matcher
    )

    image_annotator = ImageAnnotator()
    report_generator = ReportGenerator(output_folder_path=os.path.join(app.config['OUTPUT_FOLDER'], 'reports'))

    print("Models loaded successfully!")


def process_image(image_path, session_id):
    """Process a single image and return results."""
    results = image_processor.process_single_image(image_path, verbose=False)
    
    if results is None:
        return None, None
    
    if not results:
        return [], None

    # Generate annotated image
    annotated_path = None
    if results:
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'images')
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        annotated_filename = f"{session_id}_{base_name}_annotated.jpg"
        annotated_path = os.path.join(output_dir, annotated_filename)

        # Use ImageAnnotator to create annotated image
        temp_annotated = image_annotator.annotate_and_save(image_path, results, helmet_detector)
        if temp_annotated and os.path.exists(temp_annotated):
            # Move to our output folder
            import shutil
            shutil.move(temp_annotated, annotated_path)

    return results, annotated_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    # Create session ID for this batch
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(uuid.uuid4())[:8]

    all_results = []
    annotated_images = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(filepath)

            # Process image
            results, annotated_path = process_image(filepath, session_id)

            if results:
                for r in results:
                    r['annotated_image'] = os.path.basename(annotated_path) if annotated_path else ''
                all_results.extend(results)

                if annotated_path and os.path.exists(annotated_path):
                    annotated_images.append({
                        'original': filename,
                        'annotated': os.path.basename(annotated_path)
                    })

            # Clean up uploaded file
            os.remove(filepath)

    # Generate CSV report
    csv_path = None
    if all_results:
        csv_filename = f"report_{session_id}.csv"
        csv_path = os.path.join(app.config['OUTPUT_FOLDER'], 'reports', csv_filename)
        report_generator.create_report(all_results, 'csv')
        # Find the generated report (it has timestamp in name)
        reports_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'reports')
        latest_report = max(
            [f for f in os.listdir(reports_dir) if f.endswith('.csv')],
            key=lambda x: os.path.getctime(os.path.join(reports_dir, x))
        )
        csv_filename = latest_report

    # Build summary
    total_people = len(all_results)
    violations = sum(1 for r in all_results if not r['has_helmet'])
    plates_read = sum(1 for r in all_results if r['license_plate'] != 'NO_PLATE_MATCHED')

    return jsonify({
        'success': True,
        'summary': {
            'total_people': total_people,
            'violations': violations,
            'compliant': total_people - violations,
            'plates_read': plates_read
        },
        'images': annotated_images,
        'csv_report': csv_filename if all_results else None,
        'details': all_results
    })


@app.route('/output/images/<filename>')
def serve_image(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], 'images', filename),
        mimetype='image/jpeg'
    )


@app.route('/output/reports/<filename>')
def serve_report(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], 'reports', filename),
        as_attachment=True
    )


if __name__ == '__main__':
    load_models()
    print("\n" + "="*50)
    print("  Helmet Violation Detection - Web Interface")
    print("="*50)
    print("  Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)

"""
MaizeScan — Maize Leaf Disease Detection Web Application
=========================================================
Flask deployment of VGG16-based CNN trained in OOSADM Colab notebook.
Author: Brice Gaetan Nono Youmbi | ULK Data Science 2025/2026
Supervisor: Prof. Jonas Niyitegeka
"""

import os, logging, time
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from utils.leaf_image import LeafImage
from utils.cnn_model import CNNModel
from utils.diagnosis import DiagnosisResult
from utils.disease_data import DISEASE_REGISTRY, CLASS_ORDER_KEYS

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s')
log = logging.getLogger('FlaskApp')

app = Flask(__name__)
app.config['SECRET_KEY']         = os.environ.get('SECRET_KEY', 'maizescan-ulk-2026')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER']      = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = os.path.join('model', 'vgg16_maize_best.h5')
IMG_SIZE   = (224, 224)
DEMO_MODE  = False
cnn        = CNNModel(num_classes=4, img_size=IMG_SIZE)

if os.path.exists(MODEL_PATH):
    try:
        cnn.load(MODEL_PATH)
        log.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        log.warning(f"Model load failed: {e} — DEMO mode"); DEMO_MODE = True
else:
    log.warning(f"Model file not found — DEMO mode"); DEMO_MODE = True

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html', demo_mode=DEMO_MODE)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/diseases')
def diseases():
    return render_template('diseases.html', diseases=DISEASE_REGISTRY)

@app.route('/predict', methods=['POST'])
def predict():
    if 'leaf_image' not in request.files:
        return render_template('index.html', error="No file selected.", demo_mode=DEMO_MODE)
    file = request.files['leaf_image']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html', error="Please upload a JPG or PNG image.", demo_mode=DEMO_MODE)
    try:
        safe_name = f"{int(time.time())}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        file.save(save_path)
        leaf = LeafImage.from_path(save_path, img_size=IMG_SIZE)
        if DEMO_MODE:
            import numpy as np
            probs = np.array([0.05, 0.88, 0.04, 0.03])
        else:
            probs = cnn.predict(leaf.img_array)
        result   = DiagnosisResult(probs=probs, class_order=CLASS_ORDER_KEYS,
                                   registry=DISEASE_REGISTRY, leaf_image=leaf)
        response = result.build_response()
        image_url = url_for('static', filename=f"uploads/{safe_name}")
        return render_template('result.html', response=response,
                               image_url=image_url, demo_mode=DEMO_MODE)
    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        return render_template('index.html', error=f"Processing error: {str(e)}", demo_mode=DEMO_MODE)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'leaf_image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['leaf_image']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    try:
        safe_name = f"{int(time.time())}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        file.save(save_path)
        leaf  = LeafImage.from_path(save_path, img_size=IMG_SIZE)
        probs = cnn.predict(leaf.img_array) if not DEMO_MODE else __import__('numpy').array([0.05,0.88,0.04,0.03])
        result = DiagnosisResult(probs=probs, class_order=CLASS_ORDER_KEYS,
                                 registry=DISEASE_REGISTRY, leaf_image=leaf)
        return jsonify(result.build_response())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'demo_mode': DEMO_MODE, 'model_loaded': not DEMO_MODE})

@app.errorhandler(413)
def too_large(e):
    return render_template('index.html', error="File too large. Max 16 MB.", demo_mode=DEMO_MODE), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html', error="Page not found.", demo_mode=DEMO_MODE), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

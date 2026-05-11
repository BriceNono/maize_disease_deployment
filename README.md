# MaizeScan — Maize Leaf Disease Detection Web App

**Author:** Brice Gaetan Nono Youmbi (Roll No. 202211043)  
**Supervisor:** Prof. Jonas Niyitegeka  
**Institution:** Kigali Independent University ULK — Data Science Year 4, 2025/2026

## Overview

Flask-based web application deploying a VGG16 transfer learning CNN model for real-time detection of maize leaf diseases. Classifies leaf images into 4 categories:
- **Northern Leaf Blight** (*Exserohilum turcicum*)
- **Common Rust** (*Puccinia sorghi*)
- **Gray Leaf Spot** (*Cercospora zeae-maydis*)
- **Healthy**

## Project Structure

```
maizescan/
├── app.py                  ← Flask application (FlaskApp class)
├── requirements.txt        ← Python dependencies
├── Procfile                ← Gunicorn start command (Render/Heroku)
├── render.yaml             ← Render cloud deployment config
├── model/
│   ├── README.md
│   └── vgg16_maize_best.h5 ← trained model (add after training)
├── utils/
│   ├── __init__.py
│   ├── leaf_image.py       ← LeafImage class (preprocessing)
│   ├── cnn_model.py        ← CNNModel class (inference wrapper)
│   ├── diagnosis.py        ← DiagnosisResult class (response builder)
│   └── disease_data.py     ← DISEASE_REGISTRY + CLASS_ORDER_KEYS
├── templates/
│   ├── base.html           ← shared layout (nav, footer)
│   ├── index.html          ← upload / home page
│   ├── result.html         ← diagnosis result page
│   ├── diseases.html       ← disease information page
│   └── about.html          ← project info page
└── static/
    ├── css/style.css       ← all styles (dark agri theme)
    ├── js/main.js          ← nav toggle + UI helpers
    └── uploads/            ← uploaded images (auto-cleaned)
```

## Setup & Run Locally

```bash
# 1. Clone / download this folder
cd maizescan

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your trained model
#    Copy vgg16_maize_best.h5 from Google Colab to model/

# 5. Run
python app.py
# Open http://localhost:5000
```

## Deploy to Render (Free Cloud)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New Web Service → Connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. Upload `model/vgg16_maize_best.h5` via Render Disk or environment

## OOSADM Design Classes (Chapter 3)

| Class | File | Role |
|---|---|---|
| LeafImage | utils/leaf_image.py | Image validation & preprocessing |
| CNNModel | utils/cnn_model.py | VGG16 model loading & inference |
| DiagnosisResult | utils/diagnosis.py | Build full prediction response |
| FlaskApp | app.py | Route handling & orchestration |
| DiseaseClass | utils/disease_data.py | Disease metadata registry |

## API Endpoint

```
POST /api/predict
Content-Type: multipart/form-data
Body: leaf_image=<file>

Response (JSON):
{
  "prediction": "Common Rust",
  "class_key": "Common_Rust",
  "confidence": 0.88,
  "confidence_pct": "88.0%",
  "scientific": "Puccinia sorghi",
  "severity": "Moderate",
  "description": "...",
  "actions": ["...", "..."],
  "prevention": "...",
  "all_probs": [...]
}
```
# maize_disease_deployment

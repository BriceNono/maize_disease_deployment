# utils/model_loader.py
# MaizeScan — CNNModel loader
# OOSADM Role: Loads and caches the trained VGG16 model at Flask startup.
# The model is loaded ONCE (Chapter 3, Sequence Diagram architectural
# decision 2) to avoid the 10–20 s latency per request.

import os
import logging
import numpy as np

log = logging.getLogger('model_loader')

# Model is loaded as a singleton at app startup
_model = None
_model_loaded = False
_demo_mode = False   # True when model file is absent (graceful degradation)


def load_model(model_path: str):
    """
    Load the Keras .h5 model from disk and cache it globally.
    Falls back to demo mode if the file is missing
    (Activity Diagram — Step 6 decision branch, Chapter 3 §3.7).

    Args:
        model_path: path to vgg16_maize_best.h5

    Returns:
        model object or None (demo mode)
    """
    global _model, _model_loaded, _demo_mode

    if _model_loaded:
        return _model

    if not os.path.exists(model_path):
        log.warning(
            f"Model file not found at '{model_path}'. "
            "Running in DEMO MODE — predictions will be simulated."
        )
        _demo_mode = True
        _model_loaded = True
        return None

    try:
        # Deferred import to avoid TF overhead if model file missing
        import tensorflow as tf
        log.info(f"Loading VGG16 model from '{model_path}' ...")
        _model = tf.keras.models.load_model(model_path)
        _model_loaded = True
        _demo_mode = False
        log.info("Model loaded successfully.")
        return _model
    except Exception as exc:
        log.error(f"Failed to load model: {exc}")
        _demo_mode = True
        _model_loaded = True
        return None


def get_model():
    """Return the cached model (or None in demo mode)."""
    return _model


def is_demo_mode() -> bool:
    """Return True if the app is running without a real model."""
    return _demo_mode


def predict(model, img_array) -> np.ndarray:
    """
    Run model inference and return the softmax probability array.
    In demo mode, returns a plausible random probability vector.

    Args:
        model     : loaded Keras model (or None in demo mode)
        img_array : np.ndarray shape (1, 224, 224, 3)

    Returns:
        np.ndarray shape (4,) — softmax probabilities
    """
    if model is None:
        # Demo mode — simulate a high-confidence prediction for UI testing
        probs = np.array([0.02, 0.87, 0.08, 0.03], dtype=np.float32)
        return probs

    raw = model.predict(img_array, verbose=0)
    return raw[0]

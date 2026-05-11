"""
CNNModel — VGG16-based CNN wrapper for Flask inference.
Mirrors the CNNModel class from the OOSADM Colab notebook (Cell 6).
Handles model loading and single-image prediction.
"""

import logging
import numpy as np

log = logging.getLogger('CNNModel')


class CNNModel:
    """
    Wraps a loaded Keras VGG16 transfer-learning model for inference.

    Attributes:
        num_classes (int)         : number of output classes (4)
        img_size    (tuple)       : input image size (224, 224)
        model       (keras.Model) : loaded Keras model
    """

    def __init__(self, num_classes: int = 4, img_size: tuple = (224, 224)):
        self.num_classes = num_classes
        self.img_size    = img_size
        self.model       = None

    def load(self, model_path: str) -> None:
        """
        Load a saved .h5 Keras model from disk.
        Called once at Flask startup — model reused across all requests
        (Chapter 3 §3.6 Sequence Diagram, architectural decision 3).
        """
        # Import TensorFlow lazily to avoid import-time overhead
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)
        log.info(f"Model loaded: {model_path} | "
                 f"Parameters: {self.model.count_params():,}")

    def predict(self, img_array: np.ndarray) -> np.ndarray:
        """
        Run inference on a preprocessed image array.

        Args:
            img_array (np.ndarray): shape (1, 224, 224, 3), values [0,1]

        Returns:
            np.ndarray: softmax probability array, shape (4,)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        probs = self.model.predict(img_array, verbose=0)
        return probs[0]

    def get_class_count(self) -> int:
        return self.num_classes

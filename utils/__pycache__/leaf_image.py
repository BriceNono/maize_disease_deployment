"""
LeafImage — Image preprocessing utility.
Mirrors the LeafImage class from the OOSADM Colab notebook (Cell 10).
Implements FR-01: accept JPG/PNG leaf images up to 16 MB.
"""

import os
import numpy as np
from PIL import Image as PILImage


class LeafImage:
    """
    Encapsulates a single maize leaf image for CNN inference.

    Attributes:
        path      (str)        : absolute path to the image file
        img_size  (tuple)      : target (H, W) for resizing
        img_array (np.ndarray) : preprocessed array (1, H, W, 3)
        _pil_img  (PIL.Image)  : original PIL image object

    Class Methods:
        from_path(path, img_size) → LeafImage
    """

    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

    def __init__(self, path: str, img_size: tuple = (224, 224)):
        self.path      = path
        self.img_size  = img_size
        self.img_array = None
        self._pil_img  = None

    @classmethod
    def from_path(cls, path: str, img_size: tuple = (224, 224)) -> 'LeafImage':
        """
        Validate, load, and preprocess a leaf image from disk.
        Raises ValueError for unsupported formats.
        Raises FileNotFoundError if path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext not in cls.VALID_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {ext}")

        instance = cls(path=path, img_size=img_size)
        instance._load_and_preprocess()
        return instance

    def _load_and_preprocess(self) -> None:
        """
        Resize, normalise, and expand batch dimension.
        Identical pipeline to Colab notebook (Cell 10):
          1. Open with PIL and convert to RGB
          2. Resize to img_size using LANCZOS resampling
          3. Convert to float32 numpy array
          4. Normalise pixels [0,255] → [0,1]
          5. Expand dims → (1, H, W, 3)
        """
        self._pil_img  = PILImage.open(self.path).convert('RGB')
        resized        = self._pil_img.resize(self.img_size, PILImage.LANCZOS)
        arr            = np.array(resized, dtype=np.float32) / 255.0
        self.img_array = np.expand_dims(arr, axis=0)

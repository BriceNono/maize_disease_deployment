"""
DiagnosisResult — Prediction result builder.
Mirrors the DiagnosisResult class from OOSADM Colab notebook (Cell 11).
Implements FR-03 to FR-09: build and expose full diagnosis response.
"""

import numpy as np


class DiagnosisResult:
    """
    Encapsulates and formats a single CNN disease prediction.

    Attributes:
        probs        (np.ndarray) : softmax probability array (4,)
        class_order  (list)       : class key strings in output index order
        registry     (dict)       : DISEASE_REGISTRY metadata
        leaf_image               : LeafImage object (may be None)
    """

    def __init__(self, probs, class_order, registry, leaf_image=None):
        self.probs       = np.array(probs, dtype=float)
        self.class_order = class_order
        self.registry    = registry
        self.leaf_image  = leaf_image

        self._pred_idx   = int(np.argmax(self.probs))
        self._pred_key   = class_order[self._pred_idx]
        self._confidence = float(self.probs[self._pred_idx])
        self._info       = registry[self._pred_key]

    def format_confidence(self) -> str:
        """FR-04: Format confidence as percentage string."""
        return f"{self._confidence * 100:.1f}%"

    def get_all_probs_sorted(self) -> list:
        """Return list of (display_label, probability) sorted descending."""
        return sorted(
            [
                (self.registry[k]['label'], float(p))
                for k, p in zip(self.class_order, self.probs)
            ],
            key=lambda x: x[1], reverse=True
        )

    def build_response(self) -> dict:
        """
        Build the complete prediction response dictionary.
        Matches Colab DiagnosisResult.build_response() exactly.
        """
        return {
            'prediction':     self._info['label'],
            'class_key':      self._pred_key,
            'confidence':     round(self._confidence, 4),
            'confidence_pct': self.format_confidence(),
            'scientific':     self._info['scientific'],
            'severity':       self._info['severity'],
            'severity_level': self._info.get('severity_level', 0),
            'spread':         self._info['spread'],
            'color':          self._info['color'],
            'icon':           self._info.get('icon', ''),
            'description':    self._info['description'],
            'actions':        self._info['actions'],
            'prevention':     self._info['prevention'],
            'all_probs': [
                {
                    'label': self.registry[k]['label'],
                    'key':   k,
                    'prob':  round(float(p), 4),
                    'pct':   f"{float(p)*100:.1f}%",
                    'color': self.registry[k]['color'],
                }
                for k, p in zip(self.class_order, self.probs)
            ]
        }

# ====================================================================
# decide.py
# ====================================================================
# DECISION ENGINE
# --------------------------------------------------------------------
# Single 3-class Random Forest that replaces the old two-model
# pipeline (blank RF → router RF).
#
# Predicted labels:
#   1 — SKIP  : blank page → discard, no OCR
#   2 — HEAVY : degraded scan → full preprocess pipeline
#   3 — CLEAN : sharp scan → straight to OCR
# ====================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os

import cv2
import joblib
import numpy as np

# ====================================================================
# LABELS
# ====================================================================
LABEL_SKIP  = 1
LABEL_HEAVY = 2
LABEL_CLEAN = 3

LABEL_NAMES: dict[int, str] = {
    LABEL_SKIP:  "SKIP",
    LABEL_HEAVY: "HEAVY",
    LABEL_CLEAN: "CLEAN",
}

F_WHITE_RATIO   = "white_ratio"
F_STD           = "std_val"
F_COEFF         = "coeff_variation"
F_LAPLACIAN_VAR = "laplacian_var"
F_MEAN          = "mean_intensity"
F_EDGE_DENSITY  = "edge_density"

FEATURE_KEYS: list[str] = [
    F_WHITE_RATIO,
    F_STD,
    F_COEFF,
    F_LAPLACIAN_VAR,
    F_MEAN,
    F_EDGE_DENSITY,
]

TRAIN_AREA: int = 500 * 500

RECIPE_CLEAN: str = "grayscale, deskew, autocrop, qr_detect"
RECIPE_HEAVY: str = "grayscale, denoise, adaptive_threshold, deskew, autocrop, qr_detect, sharpen"


@dataclass
class DecisionResult:
    label:      int               # 1 SKIP | 2 HEAVY | 3 CLEAN
    label_name: str               # human-readable label
    confidence: float             # P(predicted class)
    recipe:     str | None        # None when label == SKIP
    probs:      dict[str, float]  # full class probability breakdown

class DecideML(ABC):
    @abstractmethod
    def predict_probabilities(self, feature_vector: np.ndarray) -> dict[int, float]:
        """Return {label_int: probability} for all classes."""


class DecideJoblib(DecideML):
    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)

    def predict_probabilities(self, feature_vector: np.ndarray) -> dict[int, float]:
        vec   = feature_vector.reshape(1, -1)
        probs = self.model.predict_proba(vec)[0]   # shape (n_classes,)
        return {
            int(cls): float(p)
            for cls, p in zip(self.model.classes_, probs)
        }


# ====================================================================
# ENGINE
# ====================================================================

class DecisionEngine:
    """
    Accepts either a ready-made DecideML provider or a file-system
    path to a joblib model.  If neither is usable the engine falls
    back to a deterministic heuristic.

    Usage
    -----
    # From a saved model file:
    engine = DecisionEngine("models/rf.joblib")

    # From an already-loaded provider:
    engine = DecisionEngine(DecideJoblib("models/rf.joblib"))
    """

    def __init__(
        self,
        provider: DecideML | str,
        skip_threshold:  float = 0.5,
        clean_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= skip_threshold <= 1.0:
            raise ValueError(
                f"skip_threshold must be in [0, 1], received {skip_threshold}"
            )
        if not 0.0 <= clean_threshold <= 1.0:
            raise ValueError(
                f"clean_threshold must be in [0, 1], received {clean_threshold}"
            )

        self.skip_threshold  = skip_threshold
        self.clean_threshold = clean_threshold

        # ── resolve provider ─────────────────────────────────────────
        if isinstance(provider, DecideML):
            self.provider: DecideML | None = provider
        elif isinstance(provider, str):
            try:
                self.provider = DecideJoblib(provider)
            except FileNotFoundError:
                print(f"ERR [DecisionEngine] model not found: {provider}")
                self.provider = None
        else:
            raise TypeError(
                f"provider must be a DecideML instance or a path string, "
                f"got {type(provider).__name__}"
            )

    # ----------------------------------------------------------------
    # Feature extraction
    # ----------------------------------------------------------------

    @staticmethod
    def extract_features(image: np.ndarray) -> dict[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.resize(gray, (500, 500), interpolation=cv2.INTER_AREA)

        std_val         = float(np.std(gray))
        mean_intensity  = float(np.mean(gray))
        background      = float(np.median(gray))
        white_ratio     = float(
            np.sum(np.abs(gray.astype(np.int32) - int(background)) < 20) / gray.size
        )
        laplacian_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edge_density    = float(np.count_nonzero(cv2.Canny(gray, 100, 200)) / gray.size)
        coeff_variation = std_val / (mean_intensity + 1e-6)

        return {
            F_WHITE_RATIO:   white_ratio,
            F_STD:           std_val,
            F_COEFF:         coeff_variation,
            F_LAPLACIAN_VAR: laplacian_var,
            F_MEAN:          mean_intensity,
            F_EDGE_DENSITY:  edge_density,
        }

    @staticmethod
    def _to_vector(features: dict[str, float]) -> np.ndarray:
        """Ordered dict → 1-D float32 array ready for model input."""
        return np.array([features[k] for k in FEATURE_KEYS], dtype=np.float32)

    # ----------------------------------------------------------------
    # Heuristic fallback (no model available)
    # ----------------------------------------------------------------

    def _fallback_result(self, features: dict[str, float]) -> DecisionResult:
        if features[F_WHITE_RATIO] > 0.95 and features[F_STD] < 15:
            label = LABEL_SKIP
        elif features[F_LAPLACIAN_VAR] < 500 and features[F_COEFF] < 0.12:
            label = LABEL_CLEAN
        else:
            label = LABEL_HEAVY

        return DecisionResult(
            label      = label,
            label_name = LABEL_NAMES[label],
            confidence = 0.0,
            recipe     = None if label == LABEL_SKIP else (
                         RECIPE_CLEAN if label == LABEL_CLEAN else RECIPE_HEAVY),
            probs      = {LABEL_NAMES[l]: 0.0
                          for l in [LABEL_SKIP, LABEL_HEAVY, LABEL_CLEAN]},
        )

    # ----------------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------------

    def evaluate(self, image: np.ndarray) -> DecisionResult:
        features = self.extract_features(image)

        if self.provider is None:
            return self._fallback_result(features)

        # 1. Prepare the vector
        vec = self._to_vector(features)

        # 2. Delegate prediction to the provider
        class_probs = self.provider.predict_probabilities(vec)

        # 3. Primary decision: argmax
        label = int(max(class_probs, key=class_probs.__getitem__))

        # 4. Threshold overrides: explicit confidence gates take priority
        if class_probs.get(LABEL_SKIP, 0.0) >= self.skip_threshold:
            label = LABEL_SKIP
        elif class_probs.get(LABEL_CLEAN, 0.0) >= self.clean_threshold:
            label = LABEL_CLEAN

        confidence = class_probs[label]
        recipe     = None if label == LABEL_SKIP else (
                     RECIPE_CLEAN if label == LABEL_CLEAN else RECIPE_HEAVY)

        return DecisionResult(
            label      = label,
            label_name = LABEL_NAMES[label],
            confidence = confidence,
            recipe     = recipe,
            probs      = {LABEL_NAMES[l]: class_probs.get(l, 0.0)
                          for l in [LABEL_SKIP, LABEL_HEAVY, LABEL_CLEAN]},
        )

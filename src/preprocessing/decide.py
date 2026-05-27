from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os

import cv2
import joblib
import numpy as np

LABEL_SKIP = 1
LABEL_HEAVY = 2
LABEL_CLEAN = 3

DEFAULT_LABEL_NAMES: dict[int, str] = {
    LABEL_SKIP: "SKIP",
    LABEL_HEAVY: "HEAVY",
    LABEL_CLEAN: "CLEAN",
}

F_WHITE_RATIO = "white_ratio"
F_STD = "std_val"
F_COEFF = "coeff_variation"
F_LAPLACIAN_VAR = "laplacian_var"
F_MEAN = "mean_intensity"
F_EDGE_DENSITY = "edge_density"

DEFAULT_FEATURE_KEYS: list[str] = [
    F_WHITE_RATIO,
    F_STD,
    F_COEFF,
    F_LAPLACIAN_VAR,
    F_MEAN,
    F_EDGE_DENSITY,
]

DEFAULT_FEATURE_EXTRACTION_CONFIG: dict = {
    "grayscale_conversion_code": "COLOR_BGR2GRAY",
    "resize_width": 500,
    "resize_height": 500,
    "resize_interpolation": "INTER_AREA",
    "white_ratio_tolerance": 20,
    "laplacian_ddepth": "CV_64F",
    "edge_density_canny_threshold1": 100,
    "edge_density_canny_threshold2": 200,
    "coeff_variation_epsilon": 1e-6,
}

DEFAULT_FALLBACK_RULES: dict = {
    "skip_white_ratio_gt": 0.95,
    "skip_std_lt": 15.0,
    "clean_laplacian_var_lt": 500.0,
    "clean_coeff_lt": 0.12,
}

DEFAULT_RECIPES: dict[int, str | None] = {
    LABEL_SKIP: None,
    LABEL_CLEAN: "grayscale, deskew, autocrop, qr_detect",
    LABEL_HEAVY: "grayscale, denoise, adaptive_threshold, deskew, autocrop, qr_detect, sharpen",
}


@dataclass
class DecisionResult:
    label: int
    label_name: str
    confidence: float
    recipe: str | None
    probs: dict[str, float]


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
        vec = feature_vector.reshape(1, -1)
        probs = self.model.predict_proba(vec)[0]
        return {
            int(cls): float(p)
            for cls, p in zip(self.model.classes_, probs)
        }


class DecisionEngine:
    def __init__(
        self,
        provider: DecideML | str | None = None,
        skip_threshold: float           = 0.5,
        clean_threshold: float          = 0.5,
        feature_extraction: dict | None = None,
        fallback_rules: dict | None     = None,
        recipes: dict | None            = None,
        label_names: dict | None        = None,
        feature_keys: list[str] | None  = None,
        missing_model_policy: str       = "fallback",
        warn_on_missing_model: bool     = True,
    ) -> None:
        self._validate_probability(skip_threshold, "skip_threshold")
        self._validate_probability(clean_threshold, "clean_threshold")

        self.skip_threshold     = skip_threshold
        self.clean_threshold = clean_threshold
        self.missing_model_policy = missing_model_policy
        self.warn_on_missing_model = warn_on_missing_model

        self.feature_cfg = {
            **DEFAULT_FEATURE_EXTRACTION_CONFIG,
            **(feature_extraction or {}),
        }
        self.fallback_rules = {
            **DEFAULT_FALLBACK_RULES,
            **(fallback_rules or {}),
        }

        raw_label_names = label_names or {}
        self.label_names: dict[int, str] = {
            int(k): v for k, v in {**DEFAULT_LABEL_NAMES, **raw_label_names}.items()
        }

        raw_recipes = recipes or {}
        normalized_recipes = {}
        for k, v in raw_recipes.items():
            normalized_recipes[int(k)] = v
        self.recipes: dict[int, str | None] = {
            **DEFAULT_RECIPES,
            **normalized_recipes,
        }

        self.feature_keys = feature_keys or list(DEFAULT_FEATURE_KEYS)

        self._validate_config()
        self.provider = self._resolve_provider(provider)

    @staticmethod
    def _validate_probability(value: float, name: str) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], received {value}")

    @staticmethod
    def _resolve_cv2_constant(name: str) -> int:
        try:
            return getattr(cv2, name)
        except AttributeError as e:
            raise ValueError(f"Unknown cv2 constant: {name}") from e

    def _validate_config(self) -> None:
        fw = self.feature_cfg["resize_width"]
        fh = self.feature_cfg["resize_height"]
        if fw <= 0 or fh <= 0:
            raise ValueError("feature_extraction.resize_width/resize_height must be > 0")

        if self.feature_cfg["white_ratio_tolerance"] < 0:
            raise ValueError("feature_extraction.white_ratio_tolerance must be >= 0")

        if self.feature_cfg["coeff_variation_epsilon"] <= 0:
            raise ValueError("feature_extraction.coeff_variation_epsilon must be > 0")

        required_fallback_keys = {
            "skip_white_ratio_gt",
            "skip_std_lt",
            "clean_laplacian_var_lt",
            "clean_coeff_lt",
        }
        missing = required_fallback_keys - set(self.fallback_rules.keys())
        if missing:
            raise ValueError(f"Missing fallback_rules keys: {sorted(missing)}")

        for label in (LABEL_SKIP, LABEL_HEAVY, LABEL_CLEAN):
            if label not in self.label_names:
                raise ValueError(f"Missing label_names entry for label {label}")
            if label not in self.recipes:
                raise ValueError(f"Missing recipes entry for label {label}")

        missing_features = [k for k in self.feature_keys if k not in DEFAULT_FEATURE_KEYS]
        if missing_features:
            raise ValueError(f"Unsupported feature_keys: {missing_features}")

        if self.missing_model_policy not in {"fallback", "raise"}:
            raise ValueError("missing_model_policy must be 'fallback' or 'raise'")

    def _resolve_provider(self, provider: DecideML | str | None) -> DecideML | None:
        if provider is None:
            return None

        if isinstance(provider, DecideML):
            return provider

        if isinstance(provider, str):
            try:
                return DecideJoblib(provider)
            except FileNotFoundError:
                if self.warn_on_missing_model:
                    print(f"ERR [DecisionEngine] model not found: {provider}")
                if self.missing_model_policy == "raise":
                    raise
                return None

        raise TypeError(
            f"provider must be a DecideML instance, path string, or None; "
            f"got {type(provider).__name__}"
        )

    def extract_features(self, image: np.ndarray) -> dict[str, float]:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        gray = image
        if image.ndim == 3:
            gray = cv2.cvtColor(
                image,
                self._resolve_cv2_constant(self.feature_cfg["grayscale_conversion_code"]),
            )

        gray = cv2.resize(
            gray,
            (self.feature_cfg["resize_width"], self.feature_cfg["resize_height"]),
            interpolation=self._resolve_cv2_constant(self.feature_cfg["resize_interpolation"]),
        )

        std_val = float(np.std(gray))
        mean_intensity = float(np.mean(gray))
        background = float(np.median(gray))

        tolerance = self.feature_cfg["white_ratio_tolerance"]
        white_ratio = float(
            np.sum(np.abs(gray.astype(np.int32) - int(background)) < tolerance) / gray.size
        )

        laplacian_var = float(
            cv2.Laplacian(
                gray,
                self._resolve_cv2_constant(self.feature_cfg["laplacian_ddepth"]),
            ).var()
        )

        edge_density = float(
            np.count_nonzero(
                cv2.Canny(
                    gray,
                    self.feature_cfg["edge_density_canny_threshold1"],
                    self.feature_cfg["edge_density_canny_threshold2"],
                )
            ) / gray.size
        )

        coeff_variation = std_val / (
            mean_intensity + self.feature_cfg["coeff_variation_epsilon"]
        )

        return {
            F_WHITE_RATIO: white_ratio,
            F_STD: std_val,
            F_COEFF: coeff_variation,
            F_LAPLACIAN_VAR: laplacian_var,
            F_MEAN: mean_intensity,
            F_EDGE_DENSITY: edge_density,
        }

    def _to_vector(self, features: dict[str, float]) -> np.ndarray:
        return np.array([features[k] for k in self.feature_keys], dtype=np.float32)

    def _recipe_for(self, label: int) -> str | None:
        return self.recipes.get(label)

    def _probs_dict(self, class_probs: dict[int, float] | None = None) -> dict[str, float]:
        class_probs = class_probs or {}
        return {
            self.label_names[label]: float(class_probs.get(label, 0.0))
            for label in (LABEL_SKIP, LABEL_HEAVY, LABEL_CLEAN)
        }

    def _fallback_result(self, features: dict[str, float]) -> DecisionResult:
        rules = self.fallback_rules

        if (
            features[F_WHITE_RATIO] > rules["skip_white_ratio_gt"]
            and features[F_STD] < rules["skip_std_lt"]
        ):
            label = LABEL_SKIP
        elif (
            features[F_LAPLACIAN_VAR] < rules["clean_laplacian_var_lt"]
            and features[F_COEFF] < rules["clean_coeff_lt"]
        ):
            label = LABEL_CLEAN
        else:
            label = LABEL_HEAVY

        return DecisionResult(
            label       = label,
            label_name  = self.label_names[label],
            confidence  = 0.0,
            recipe      = self._recipe_for(label),
            probs       = self._probs_dict(),
        )

    def evaluate(self, image: np.ndarray) -> DecisionResult:
        features = self.extract_features(image)

        if self.provider is None:
            return self._fallback_result(features)

        vec = self._to_vector(features)
        class_probs = self.provider.predict_probabilities(vec)

        label = int(max(class_probs, key=class_probs.__getitem__))

        if class_probs.get(LABEL_SKIP, 0.0) >= self.skip_threshold:
            label = LABEL_SKIP
        elif class_probs.get(LABEL_CLEAN, 0.0) >= self.clean_threshold:
            label = LABEL_CLEAN

        confidence = float(class_probs.get(label, 0.0))

        return DecisionResult(
            label       = label,
            label_name  = self.label_names[label],
            confidence  = confidence,
            recipe      = self._recipe_for(label),
            probs       = self._probs_dict(class_probs),
        )
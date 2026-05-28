from __future__ import annotations

from abc        import ABC, abstractmethod
import os

import joblib
import numpy    as np

from constants  import (
    FALLBACK_RULES, 
    FEATURE_KEYS, 
    LABEL_CLEAN, 
    LABEL_HEAVY, 
    LABEL_NAMES, 
    LABEL_SKIP, 
    RECIPES)
from features   import extract_features, to_vector
from models     import DecisionResult


class DecideML(ABC):
    @abstractmethod
    def predict_probabilities(self, feature_vector: np.ndarray) -> dict[int, float]:
        raise NotImplementedError


class DecideJoblib(DecideML):
    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)

    def predict_probabilities(self, feature_vector: np.ndarray) -> dict[int, float]:
        vector = feature_vector.reshape(1, -1)
        probs = self.model.predict_proba(vector)[0]
        return {int(cls): float(prob)
                for cls, prob in zip(self.model.classes_, probs)}


class DecisionEngine:
    def __init__(
        self,
        provider: DecideML | str | None = None,
        skip_threshold: float = 0.5,
        clean_threshold: float = 0.5,
        missing_model_policy: str = "fallback",
        warn_on_missing_model: bool = True,
    ) -> None:
        self.skip_threshold = skip_threshold
        self.clean_threshold = clean_threshold
        self.missing_model_policy = missing_model_policy
        self.warn_on_missing_model = warn_on_missing_model
        self.feature_keys = FEATURE_KEYS
        self.provider = self._resolve_provider(provider)

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
                    print(f"ERR: DecisionEngine model not found: {provider}")
                if self.missing_model_policy == "raise":
                    raise
                return None
        raise TypeError("provider must be a DecideML instance, path string, or None")

    def _fallback_result(self, features: dict[str, float]) -> DecisionResult:
        if (
            features["white_ratio"] > FALLBACK_RULES["skip_white_ratio_gt"]
            and features["std_val"] < FALLBACK_RULES["skip_std_lt"]
        ):
            label = LABEL_SKIP
        elif (
            features["laplacian_var"] < FALLBACK_RULES["clean_laplacian_var_lt"]
            and features["coeff_variation"] < FALLBACK_RULES["clean_coeff_lt"]
        ):
            label = LABEL_CLEAN
        else:
            label = LABEL_HEAVY

        return DecisionResult(
            label       = label,
            label_name  = LABEL_NAMES[label],
            confidence  = 0.0,
            recipe      = RECIPES[label],
            probs       = {name: 0.0 for name in LABEL_NAMES.values()},
        )

    def evaluate(self, image: np.ndarray) -> DecisionResult:
        features = extract_features(image)

        if self.provider is None:
            return self._fallback_result(features)

        vector = to_vector(features, self.feature_keys)
        class_probs = self.provider.predict_probabilities(vector)
        label = int(max(class_probs, key=class_probs.get))

        if class_probs.get(LABEL_SKIP, 0.0) >= self.skip_threshold:
            label = LABEL_SKIP
        elif class_probs.get(LABEL_CLEAN, 0.0) >= self.clean_threshold:
            label = LABEL_CLEAN

        return DecisionResult(
            label       = label,
            label_name  = LABEL_NAMES[label],
            confidence  = float(class_probs.get(label, 0.0)),
            recipe      = RECIPES[label],
            probs       = {LABEL_NAMES[k]: float(v) for k, v in class_probs.items()},
        )

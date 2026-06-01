from __future__ import annotations

import numpy    as np

from .constants import (LABEL_CLEAN, LABEL_HEAVY, 
                        LABEL_SKIP, STEP_PARAMS)
from .decision  import DecisionEngine
from .models    import PreprocessResult
from .steps     import *

STEP_MAP = {
    "grayscale": to_grayscale,
    "orientation": orient,
    "perspective": perspective_correct,
    "denoise": denoise,
    "deskew": deskew,
    "autocrop": autocrop,
    "adaptive_threshold": adaptive_threshold,
    "sharpen": sharpen,
    "enhance_contrast": enhance_contrast,
    "levels": levels,
    "qr_detect": qr_detect
}

RECIPES = {
    LABEL_SKIP: [],
    LABEL_CLEAN: ["grayscale", "denoise", "levels", "deskew", "autocrop"],
    LABEL_HEAVY: ["grayscale", "denoise", "levels", "adaptive_threshold", "deskew", "autocrop"],
}


class Preprocessing:
    def __init__(self, provider: str | None = None) -> None:
        self.engine = DecisionEngine(provider=provider)
        self.qr_buffer: list = []

    def apply_recipe(
            self,
            image: np.ndarray,
            recipe: list[str],
            label: int) -> np.ndarray:

        current = image
        params = STEP_PARAMS.get(label, {})
        for step_name in recipe:
            step = STEP_MAP[step_name]
            kwargs = params.get(step_name, {})
            current = step(current, **kwargs)

        # Only here, on a NumPy image
        current, self.qr_buffer = qr_detect(current)
        return current

    def process(self, image: np.ndarray) -> PreprocessResult:
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("Input image is empty.")

        working = image.copy()
        decision = self.engine.evaluate(working)
        processed = self.apply_recipe(working, RECIPES[decision.label], decision.label)

        metadata = {
            "status": decision.label_name.lower(),
            "qrcodes": self.qr_buffer,
        }

        return PreprocessResult(
            image    = processed,
            metadata = metadata,
            decision = decision,
        )

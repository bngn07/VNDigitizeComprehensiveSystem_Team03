from __future__ import annotations

import numpy    as np

from constants import LABEL_CLEAN, LABEL_HEAVY, LABEL_SKIP
from decision  import DecisionEngine
from models    import PreprocessResult
from steps     import (adaptive_threshold, 
                        autocrop, 
                        denoise, 
                        deskew, 
                        orient, 
                        perspective_correct, 
                        qr_detect, 
                        sharpen, 
                        to_grayscale)


STEP_MAP = {
    "grayscale": to_grayscale,
    "orientation": orient,
    "perspective": perspective_correct,
    "denoise": denoise,
    "deskew": deskew,
    "autocrop": autocrop,
    "adaptive_threshold": adaptive_threshold,
    "sharpen": sharpen,
}

RECIPES = {
    LABEL_SKIP: [],
    LABEL_CLEAN: ["grayscale", "deskew", "autocrop"],
    LABEL_HEAVY: ["grayscale", "denoise", "adaptive_threshold", "deskew", "autocrop", "sharpen"],
}


class Preprocessing:
    def __init__(self, provider: str | None = None) -> None:
        self.engine = DecisionEngine(provider=provider)
        self.qr_buffer: list = []

    def apply_recipe(self, image: np.ndarray, recipe: list[str]) -> np.ndarray:
        current = image
        for step_name in recipe:
            step = STEP_MAP[step_name]
            current = step(current)
        current, self.qr_buffer = qr_detect(current)
        return current

    def process(self, image: np.ndarray) -> PreprocessResult:
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("Input image is empty.")

        working = image.copy()
        decision = self.engine.evaluate(working)
        processed = self.apply_recipe(working, RECIPES[decision.label])

        metadata = {
            "status": decision.label_name.lower(),
            "qrcodes": self.qr_buffer,
        }

        return PreprocessResult(
            image    = processed,
            metadata = metadata,
            decision = decision,
        )

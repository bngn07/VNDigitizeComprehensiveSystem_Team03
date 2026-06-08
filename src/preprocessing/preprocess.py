from __future__ import annotations

import cv2
import magic
import numpy as np

from .constants import (LABEL_CLEAN, LABEL_HEAVY,
                        LABEL_SKIP, STEP_PARAMS, PreprocessError, RECIPES, Step)
from .decision  import DecisionEngine
from .models    import PreprocessResult, CodeResult
from .steps import (
    adaptive_threshold,
    autocrop,
    denoise,
    deskew,
    enhance_contrast,
    levels,
    orient,
    perspective_correct,
    qr_detect,
    sharpen,
    to_grayscale,
)

STEP_MAP = {
    Step.GRAYSCALE: to_grayscale,
    Step.ORIENTATION: orient,
    Step.PERSPECTIVE: perspective_correct,
    Step.DENOISE: denoise,
    Step.DESKEW: deskew,
    Step.AUTOCROP: autocrop,
    Step.ADAPTIVE_THRESHOLD: adaptive_threshold,
    Step.SHARPEN: sharpen,
    Step.ENHANCE_CONTRAST: enhance_contrast,
    Step.LEVELS: levels,
    Step.QR_DETECT: qr_detect
}

class Preprocessing:
    def __init__(self, provider: str | None = None) -> None:
        self.engine = DecisionEngine(provider=provider)
        self.qr_buffer: list[CodeResult] = []

    def apply_recipe(
        self,
        image: np.ndarray,
        recipe: list[str],
        label: int,
    ) -> np.ndarray:
        current = image
        params = STEP_PARAMS.get(label, {})

        for step_name in recipe:
            step = STEP_MAP[step_name]
            kwargs = params.get(step_name, {})
            current = step(current, **kwargs)

        current, self.qr_buffer = qr_detect(current)
        return current

    @staticmethod
    def is_image(file_path: str) -> bool:
        try:
            true_mime = magic.from_file(file_path, mime=True)
        except Exception as e:
            raise ValueError(PreprocessError.FILE_TYPE_UNKNOWN.format(detail=e))

        valid_image_mimes = {
            "image/jpeg", "image/png", "image/bmp",
            "image/tiff", "image/gif", "image/webp"
        }

        if true_mime in valid_image_mimes:
            return True

        if true_mime == "application/pdf":
            raise ValueError(PreprocessError.FILE_TYPE_PDF)

        if true_mime.startswith("video/"):
            raise ValueError(PreprocessError.FILE_TYPE_VIDEO.format(detail=true_mime))

        raise ValueError(PreprocessError.FILE_TYPE_UNSUPPORTED.format(detail=true_mime))

    def process(self, file_path: str) -> PreprocessResult:
        self.is_image(file_path)

        image = cv2.imread(file_path)
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError(PreprocessError.CORRUPTED_IMAGE)

        working = image.copy()
        decision = self.engine.evaluate(working)
        processed = self.apply_recipe(
            working,
            RECIPES[decision.label],
            decision.label,
        )

        metadata = {
            "status": decision.label_name.lower(),
            "qrcodes": self.qr_buffer,
        }

        return PreprocessResult(
            image    = processed,
            metadata = metadata,
            decision = decision,
        )
from __future__     import annotations

from dataclasses    import dataclass

import cv2
import numpy as np
import pytesseract
from pytesseract    import Output

from models         import RotationResult


@dataclass
class RotationConfig:
    lang: str | None = None
    min_confidence: float = 0.1
    apply_rotation: bool = True


class RotationDetector:
    def __init__(self, config: RotationConfig | None = None) -> None:
        self.config = config or RotationConfig()

    def detect(self, image: np.ndarray) -> RotationResult:
        if image is None or image.size == 0:
            return RotationResult(angle=0.0, confidence=0.0)

        try:
            osd_data = pytesseract.image_to_osd(
                image,
                lang = self.config.lang,
                config = "--psm 0",
                output_type = Output.DICT,
            )
        except Exception:
            return RotationResult(angle = 0.0, confidence = 0.0)

        angle = float(osd_data.get("rotate", 0.0))
        confidence = float(osd_data.get("orientation_conf", 0.0)) / 100.0
        script = str(osd_data.get("script", "Unknown"))
        return RotationResult(angle = angle, confidence = confidence, script = script)

    def correct(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        result = self.detect(image)
        if result.confidence < self.config.min_confidence:
            return image
        if not self.config.apply_rotation or result.angle == 0:
            return image

        mapping = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        rotate_code = mapping.get(int(result.angle))
        if rotate_code is None:
            return image

        return cv2.rotate(image, rotate_code)

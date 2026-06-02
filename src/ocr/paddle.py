from __future__     import annotations

from dataclasses    import asdict, dataclass
from typing         import Any

import cv2
import numpy as np
from paddleocr      import PaddleOCR

from .base          import BaseOCR
from .models        import OCRBlockResult, TextBlock


@dataclass(frozen=True)
class PaddleConfig:
    lang: str = "en"
    device: str = "cpu"
    ocr_version: str = "PP-OCRv5"
    use_textline_orientation: bool = False
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    det_model_dir: str | None = None
    rec_model_dir: str | None = None


class PaddleOCRModel:
    def __init__(self, config: PaddleConfig | None = None) -> None:
        self.config = config or PaddleConfig()
        kwargs = {key: value for key, value in asdict(self.config).items() if value is not None}
        self.engine = PaddleOCR(**kwargs)


class PaddleTextRecognizer(BaseOCR):
    def __init__(self, model: PaddleOCRModel | None = None) -> None:
        self.model = model or PaddleOCRModel()

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        if image is None or getattr(image, "size", 0) == 0:
            return image
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return image

    @staticmethod
    def _parse_result(raw: Any) -> OCRBlockResult:
        if not raw or not isinstance(raw, list) or not raw[0]:
            return OCRBlockResult(blocks=[])

        blocks: list[TextBlock] = []
        for line in raw[0]:
            if not line or len(line) < 2:
                continue

            polygon = line[0]
            text_data = line[1]

            if isinstance(text_data, (tuple, list)) and len(text_data) == 2:
                text = str(text_data[0]).strip()
                confidence = float(text_data[1])
            elif isinstance(text_data, str):
                text = text_data.strip()
                confidence = float(line[2]) if len(line) > 2 else 0.0
            else:
                continue

            if not text:
                continue

            blocks.append(
                TextBlock(
                    text=text,
                    polygon=[(int(point[0]), int(point[1])) for point in polygon],
                    confidence=confidence,
                )
            )

        return OCRBlockResult(blocks=blocks)

    def _recognize_single(self, image: np.ndarray) -> OCRBlockResult:
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("Input image is empty.")

        rgb = self._to_rgb(image)
        raw = self.model.engine.ocr(rgb)
        return self._parse_result(raw)

    def recognize(self, image: np.ndarray | list[np.ndarray]) -> OCRBlockResult | list[OCRBlockResult]:
        if isinstance(image, list):
            return [self._recognize_single(item) for item in image]
        return self._recognize_single(image)
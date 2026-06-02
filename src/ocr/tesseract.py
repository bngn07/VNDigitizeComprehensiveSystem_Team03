from __future__     import annotations

from dataclasses    import dataclass
from typing         import Any

import numpy as np
import pytesseract

from .base          import BaseOCR
from .models        import OCRBlockResult, TextBlock


@dataclass(frozen = True)
class TesseractConfig:
    language: str = "vie"
    page_segmentation_mode: int = 3
    ocr_engine_mode: int = 3
    extra_config: str = ""
    nice: int = 0
    timeout: float = 0.0


class TesseractOCR(BaseOCR):
    def __init__(self, config: TesseractConfig | None = None) -> None:
        self.config = config or TesseractConfig()

    def _build_config_string(self) -> str:
        parts = [
            f"--oem {self.config.ocr_engine_mode}",
            f"--psm {self.config.page_segmentation_mode}",
        ]
        if self.config.extra_config.strip():
            parts.append(self.config.extra_config.strip())
        return " ".join(parts)

    @staticmethod
    def _normalize_confidence(raw_confidence: Any) -> float:
        if str(raw_confidence) in {"-1", "nan", "None"}:
            return 0.0
        try:
            return float(raw_confidence) / 100.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _bbox_to_polygon(
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> list[tuple[int, int]]:
        return [
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
        ]

    def _parse_data(self, data: dict[str, list[Any]]) -> OCRBlockResult:
        blocks: list[TextBlock] = []
        rows = zip(
            data["text"],
            data["conf"],
            data["left"],
            data["top"],
            data["width"],
            data["height"],
        )

        for text, conf, left, top, width, height in rows:
            normalized_text = text.strip()
            if not normalized_text:
                continue

            blocks.append(
                TextBlock(
                    text = normalized_text,
                    polygon = self._bbox_to_polygon(int(left), int(top), int(width), int(height)),
                    confidence = self._normalize_confidence(conf),
                )
            )

        return OCRBlockResult(blocks=blocks)

    def _recognize_single(self, image: np.ndarray) -> OCRBlockResult:
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("Input image is empty.")

        data = pytesseract.image_to_data(
            image,
            lang        = self.config.language,
            config      = self._build_config_string(),
            output_type = pytesseract.Output.DICT,
            nice        = self.config.nice,
            timeout     = self.config.timeout,
        )
        return self._parse_data(data)

    def recognize(self, image: np.ndarray | list[np.ndarray]) -> OCRBlockResult | list[OCRBlockResult]:
        if isinstance(image, list):
            return [self._recognize_single(item) for item in image]
        return self._recognize_single(image)

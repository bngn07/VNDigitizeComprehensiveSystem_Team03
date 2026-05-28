from __future__ import annotations

from pyzbar.pyzbar import decode
import numpy as np

from models import CodeResult


class CodeDetector:
    def __init__(self, types: list | None = None, bbox_format: str = "ltwh") -> None:
        self.types = types
        self.bbox_format = bbox_format

    def _build_bbox(self, result) -> tuple[int, int, int, int]:
        left = result.rect.left
        top = result.rect.top
        width = result.rect.width
        height = result.rect.height

        if self.bbox_format == "xyxy":
            return left, top, left + width, top + height
        return left, top, width, height

    def detect(self, image: np.ndarray) -> list[CodeResult]:
        if image is None or image.size == 0:
            return []

        results = decode(image)
        parsed: list[CodeResult] = []

        for item in results:
            parsed.append(
                CodeResult(
                    type=item.type,
                    content=item.data.decode("utf-8", errors="replace"),
                    bbox=self._build_bbox(item),
                    polygon=[(point.x, point.y) for point in item.polygon],
                    quality=getattr(item, "quality", None),
                    orientation=getattr(item, "orientation", None),
                )
            )

        return parsed

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol


DEFAULT_CODE_DETECTOR_CONFIG: dict = {
    "types": None,
    "empty_image_policy": "return_empty",
    "invalid_types_policy": "raise",
    "decode_encoding": "utf-8",
    "decode_errors": "replace",
    "bbox_format": "ltwh",
    "include_polygon": True,
    "include_quality": True,
    "include_orientation": True,
    "include_raw_bytes": False,
    "dedupe_by_content": False,
}


@dataclass
class CodeResult:
    type: str
    content: str
    bbox: tuple[int, int, int, int]
    polygon: list[tuple[int, int]] | None = None
    quality: int | None = None
    orientation: str | None = None
    raw_bytes: bytes | None = None


class CodeDetector:
    def __init__(self, **config: Any):
        self.config = {
            **DEFAULT_CODE_DETECTOR_CONFIG,
            **config,
        }
        self._validate_config()
        self._types = self._normalize_types(self.config["types"])

    def _validate_config(self) -> None:
        if self.config["empty_image_policy"] not in {"return_empty", "raise"}:
            raise ValueError(
                "empty_image_policy must be 'return_empty' or 'raise'"
            )

        if self.config["invalid_types_policy"] not in {"raise", "ignore"}:
            raise ValueError(
                "invalid_types_policy must be 'raise' or 'ignore'"
            )

        if self.config["bbox_format"] not in {"ltwh", "xyxy"}:
            raise ValueError(
                "bbox_format must be 'ltwh' or 'xyxy'"
            )

    def _normalize_types(self, types: list[str] | None) -> list[str] | None:
        if types is None:
            return None

        valid = set(ZBarSymbol.__members__.keys())
        invalid = sorted(set(types) - valid)

        if invalid:
            if self.config["invalid_types_policy"] == "raise":
                raise ValueError(
                    f"Unknown code types: {invalid}. Valid types: {sorted(valid)}"
                )
            types = [t for t in types if t in valid]

        return types

    def _build_bbox(self, result: Any) -> tuple[int, int, int, int]:
        left = result.rect.left
        top = result.rect.top
        width = result.rect.width
        height = result.rect.height

        if self.config["bbox_format"] == "xyxy":
            return (left, top, left + width, top + height)

        return (left, top, width, height)

    def _parse(self, result: Any) -> CodeResult:
        content = result.data.decode(
            self.config["decode_encoding"],
            errors=self.config["decode_errors"],
        )

        polygon = None
        if self.config["include_polygon"]:
            polygon = [(p.x, p.y) for p in result.polygon]

        quality = result.quality if self.config["include_quality"] else None

        orientation = None
        if self.config["include_orientation"]:
            orientation = getattr(result, "orientation", None)

        raw_bytes = result.data if self.config["include_raw_bytes"] else None

        return CodeResult(
            type=result.type,
            content=content,
            bbox=self._build_bbox(result),
            polygon=polygon,
            quality=quality,
            orientation=orientation,
            raw_bytes=raw_bytes,
        )

    def _handle_empty_image(self) -> list[CodeResult]:
        if self.config["empty_image_policy"] == "raise":
            raise ValueError("Input image is empty.")
        return []

    def detect(self, image: np.ndarray) -> list[CodeResult]:
        if image is None or image.size == 0:
            return self._handle_empty_image()

        symbols = None
        if self._types:
            symbols = [ZBarSymbol[t] for t in self._types]

        results = decode(image = image, symbols = symbols) if symbols else decode(image)
        parsed = [self._parse(r) for r in results]

        if self.config["dedupe_by_content"]:
            deduped: list[CodeResult] = []
            seen: set[tuple[str, str]] = set()
            for item in parsed:
                key = (item.type, item.content)
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)
            return deduped

        return parsed
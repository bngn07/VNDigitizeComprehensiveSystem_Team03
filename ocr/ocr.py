from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, overload

import numpy as np


DEFAULT_OCR_CORE_CONFIG: dict = {
    "bbox_format": "ltwh",
    "empty_texts_confidence": 0.0,
    "raw_string_confidence": 0.0,
    "aggregate_confidence": "mean",
    "json_encoding": "utf-8",
    "json_ensure_ascii": False,
    "json_indent": None,
    "string_repr": {
        "truncate_text": True,
        "max_text_length": 40,
        "show_index": True,
        "show_confidence": True,
        "show_bbox": True,
        "header_width": 50,
        "title": " OCR INFERENCE RESULT ",
        "raw_string_label": "Raw String",
    },
}


def _deep_merge(base: dict, override: dict | None) -> dict:
    if not override:
        return dict(base)

    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class OCRCoreConfig:
    _config: dict = dict(DEFAULT_OCR_CORE_CONFIG)

    @classmethod
    def set(cls, config: dict | None = None) -> None:
        cls._config = _deep_merge(DEFAULT_OCR_CORE_CONFIG, config or {})
        cls._validate()

    @classmethod
    def get(cls) -> dict:
        return cls._config

    @classmethod
    def _validate(cls) -> None:
        cfg = cls._config
        repr_cfg = cfg["string_repr"]

        if cfg["bbox_format"] not in {"ltwh", "xyxy"}:
            raise ValueError("bbox_format must be 'ltwh' or 'xyxy'")

        if cfg["aggregate_confidence"] not in {"mean", "max", "sum"}:
            raise ValueError("aggregate_confidence must be 'mean', 'max', or 'sum'")

        if repr_cfg["max_text_length"] <= 0:
            raise ValueError("string_repr.max_text_length must be > 0")

        if repr_cfg["header_width"] <= 0:
            raise ValueError("string_repr.header_width must be > 0")

# @dataclass(frozen=True)
@dataclass()
class TextBlock:
    text: str
    bounding_polygon: list[tuple[int, int]]
    confidence: float

    def bounding_box(self) -> tuple[int, int, int, int]:
        if not self.bounding_polygon:
            return (0, 0, 0, 0)

        xs = [p[0] for p in self.bounding_polygon]
        ys = [p[1] for p in self.bounding_polygon]

        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)

        bbox_format = OCRCoreConfig.get()["bbox_format"]
        if bbox_format == "xyxy":
            return (min_x, min_y, max_x, max_y)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _display_text(self) -> str:
        cfg = OCRCoreConfig.get()["string_repr"]

        if not cfg["truncate_text"]:
            return self.text

        max_len = cfg["max_text_length"]
        if len(self.text) <= max_len:
            return self.text

        if max_len <= 3:
            return self.text[:max_len]

        return self.text[: max_len - 3] + "..."

    def __str__(self) -> str:
        cfg = OCRCoreConfig.get()["string_repr"]
        parts = []

        if cfg.get("show_confidence", True):
            parts.append(f"[Conf: {self.confidence:.4f}]")

        parts.append(f"\"{self._display_text()}\"")

        if cfg.get("show_bbox", True):
            parts.append(f"| Box: {self.bounding_box()}")

        return " ".join(parts)


# @dataclass(frozen=True)
@dataclass()
class OCRResult:
    texts: list[TextBlock] | str

    @property
    def confidence(self) -> float:
        cfg = OCRCoreConfig.get()

        if isinstance(self.texts, str):
            return float(cfg["raw_string_confidence"])

        if not self.texts:
            return float(cfg["empty_texts_confidence"])

        values = [block.confidence for block in self.texts]
        mode = cfg["aggregate_confidence"]

        if mode == "max":
            return max(values)

        if mode == "sum":
            return sum(values)

        return sum(values) / len(values)

    def to_dict(self) -> dict[str, Any]:
        data_dict = asdict(self)
        data_dict["confidence"] = self.confidence
        return data_dict

    def to_json(self, output_path: str | None = None) -> str:
        cfg = OCRCoreConfig.get()

        json_string = json.dumps(
            self.to_dict(),
            ensure_ascii=cfg["json_ensure_ascii"],
            indent=cfg["json_indent"],
        )

        if output_path:
            with open(output_path, "w", encoding=cfg["json_encoding"]) as f:
                f.write(json_string)
            print(f"output -> {output_path}")

        return json_string

    def __str__(self) -> str:
        cfg = OCRCoreConfig.get()
        repr_cfg = cfg["string_repr"]
        width = repr_cfg["header_width"]

        lines = ["=" * width]
        lines.append(repr_cfg["title"])
        lines.append("=" * width)

        if isinstance(self.texts, str):
            lines.append(f" Output Type : {repr_cfg['raw_string_label']}")
            lines.append(f" Confidence : {self.confidence:.4f}")
            lines.append("-" * width)
            lines.append(f" {self.texts}")
            lines.append("=" * width)
            return "\n".join(lines)

        lines.append(f" Total Blocks : {len(self.texts)}")
        lines.append(f" Confidence : {self.confidence:.4f}")
        lines.append("-" * width)

        show_index = repr_cfg.get("show_index", True)
        for i, block in enumerate(self.texts):
            prefix = f" [{i:02d}] " if show_index else " "
            lines.append(f"{prefix}{block}")

        lines.append("=" * width)
        return "\n".join(lines)


class BaseOCR(ABC):
    @overload
    def recognize(self, image: np.ndarray) -> OCRResult: ...

    @overload
    def recognize(self, image: list[np.ndarray]) -> list[OCRResult]: ...

    @abstractmethod
    def recognize(
        self,
        image: np.ndarray | list[np.ndarray],
    ) -> OCRResult | list[OCRResult]:
        pass
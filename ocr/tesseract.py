from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytesseract

from .ocr import BaseOCR, OCRResult, TextBlock


DEFAULT_TESSERACT_CONFIG: dict = {
    "engine": {
        "language": "vie",
        "page_segmentation_mode": 3,
        "ocr_engine_mode": 3,
        "extra_config": "",
        "output_type": "DICT",
        "nice": 0,
        "timeout": 0,
    },
    "input": {
        "empty_image_policy": "raise",
    },
    "parse": {
        "text_key": "text",
        "confidence_key": "conf",
        "left_key": "left",
        "top_key": "top",
        "width_key": "width",
        "height_key": "height",
        "skip_empty_text": True,
        "strip_text": True,
        "invalid_confidence_values": ["-1"],
        "confidence_divisor": 100.0,
        "fallback_confidence": 0.0,
        "bbox_to_polygon": True,
    },
    "runtime": {
        "batch_mode": "loop",
        "error_policy": "raise",
        "warn_on_error": True,
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


@dataclass(frozen=True)
class TesseractParams:
    language: str = "vie"
    page_segmentation_mode: int = 3
    ocr_engine_mode: int = 3
    extra_config: str = ""
    output_type: str = "DICT"
    nice: int = 0
    timeout: float = 0.0


class TesseractModel:
    def __init__(self, params: TesseractParams | None = None):
        self.params = params or TesseractParams()
        self.config_string = self._build_config_string()

    def _build_config_string(self) -> str:
        parts = [
            f"--oem {self.params.ocr_engine_mode}",
            f"--psm {self.params.page_segmentation_mode}",
        ]

        extra = self.params.extra_config.strip()
        if extra:
            parts.append(extra)

        return " ".join(parts)


class TesseractOCR(BaseOCR):
    def __init__(
        self,
        model: TesseractModel | None = None,
        config: dict | None = None,
    ):
        super().__init__()
        self.config = _deep_merge(DEFAULT_TESSERACT_CONFIG, config or {})

        model_cfg = self.config["engine"]
        self._validate_config()

        self.model = model or TesseractModel(
            TesseractParams(
                language=model_cfg["language"],
                page_segmentation_mode=model_cfg["page_segmentation_mode"],
                ocr_engine_mode=model_cfg["ocr_engine_mode"],
                extra_config=model_cfg.get("extra_config", ""),
                output_type=model_cfg.get("output_type", "DICT"),
                nice=model_cfg.get("nice", 0),
                timeout=float(model_cfg.get("timeout", 0) or 0),
            )
        )

    def _validate_config(self) -> None:
        engine_cfg = self.config["engine"]
        input_cfg = self.config["input"]
        parse_cfg = self.config["parse"]
        runtime_cfg = self.config["runtime"]

        if input_cfg["empty_image_policy"] not in {"raise", "return_empty"}:
            raise ValueError(
                "input.empty_image_policy must be 'raise' or 'return_empty'"
            )

        if runtime_cfg["batch_mode"] not in {"loop"}:
            raise ValueError("runtime.batch_mode currently only supports 'loop'")

        if runtime_cfg["error_policy"] not in {"raise", "return_empty"}:
            raise ValueError(
                "runtime.error_policy must be 'raise' or 'return_empty'"
            )

        if engine_cfg["output_type"] != "DICT":
            raise ValueError("engine.output_type currently only supports 'DICT'")

        if engine_cfg["page_segmentation_mode"] < 0:
            raise ValueError("engine.page_segmentation_mode must be >= 0")

        if engine_cfg["ocr_engine_mode"] < 0:
            raise ValueError("engine.ocr_engine_mode must be >= 0")

        if parse_cfg["confidence_divisor"] <= 0:
            raise ValueError("parse.confidence_divisor must be > 0")

        required_parse_keys = {
            "text_key",
            "confidence_key",
            "left_key",
            "top_key",
            "width_key",
            "height_key",
        }
        missing = required_parse_keys - set(parse_cfg.keys())
        if missing:
            raise ValueError(f"Missing parse config keys: {sorted(missing)}")

    def _warn(self, message: str) -> None:
        print(message)

    def _handle_empty_image(self) -> OCRResult:
        if self.config["input"]["empty_image_policy"] == "raise":
            raise ValueError("Input image is empty.")
        return OCRResult(texts=[])

    def _handle_runtime_error(self, exc: Exception) -> OCRResult:
        if self.config["runtime"].get("warn_on_error", True):
            self._warn(f"WARN [TesseractOCR] {exc}")

        if self.config["runtime"]["error_policy"] == "raise":
            raise

        return OCRResult(texts=[])

    def _build_tesseract_kwargs(self, image: np.ndarray) -> dict[str, Any]:
        kwargs = {
            "image": image,
            "lang": self.model.params.language,
            "config": self.model.config_string,
            "output_type": pytesseract.Output.DICT,
        }

        if self.model.params.nice:
            kwargs["nice"] = self.model.params.nice

        if self.model.params.timeout:
            kwargs["timeout"] = self.model.params.timeout

        return kwargs

    def _normalize_confidence(self, raw_conf: Any) -> float:
        parse_cfg = self.config["parse"]

        if str(raw_conf) in set(parse_cfg["invalid_confidence_values"]):
            return float(parse_cfg["fallback_confidence"])

        try:
            return float(raw_conf) / float(parse_cfg["confidence_divisor"])
        except (TypeError, ValueError):
            return float(parse_cfg["fallback_confidence"])

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

    def _parse_data_result(self, data: dict) -> OCRResult:
        parse_cfg = self.config["parse"]
        texts = []

        rows = zip(
            data[parse_cfg["text_key"]],
            data[parse_cfg["confidence_key"]],
            data[parse_cfg["left_key"]],
            data[parse_cfg["top_key"]],
            data[parse_cfg["width_key"]],
            data[parse_cfg["height_key"]],
        )

        for text, conf, left, top, width, height in rows:
            normalized_text = text.strip() if parse_cfg.get("strip_text", True) else text

            if parse_cfg.get("skip_empty_text", True) and normalized_text == "":
                continue

            if not normalized_text:
                continue

            confidence = self._normalize_confidence(conf)

            if parse_cfg.get("bbox_to_polygon", True):
                polygon = self._bbox_to_polygon(
                    int(left),
                    int(top),
                    int(width),
                    int(height),
                )
            else:
                polygon = []

            text_block = TextBlock(
                text=normalized_text,
                bounding_polygon=polygon,
                confidence=confidence,
            )
            texts.append(text_block)

        return OCRResult(texts=texts)

    def _recognize_single(self, image: np.ndarray) -> OCRResult:
        if image is None or getattr(image, "size", 0) == 0:
            return self._handle_empty_image()

        try:
            data = pytesseract.image_to_data(**self._build_tesseract_kwargs(image))
            return self._parse_data_result(data)
        except Exception as exc:
            return self._handle_runtime_error(exc)

    def _recognize_batch(self, images: list[np.ndarray]) -> list[OCRResult]:
        if self.config["runtime"]["batch_mode"] == "loop":
            return [self._recognize_single(img) for img in images]

        raise ValueError(
            f"Unsupported runtime.batch_mode: {self.config['runtime']['batch_mode']}"
        )

    def recognize(
        self,
        image: np.ndarray | list[np.ndarray],
    ) -> OCRResult | list[OCRResult]:
        if isinstance(image, list):
            return self._recognize_batch(image)
        return self._recognize_single(image)
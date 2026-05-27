from __future__ import annotations

# -- built in
from dataclasses import dataclass, asdict
from typing import Any

# -- 3rd party
import numpy as np
import cv2
from paddleocr import PaddleOCR

# -- own
from .ocr import BaseOCR, OCRResult, TextBlock


DEFAULT_PADDLE_CONFIG: dict = {
    "engine": {
        # "lang": "en",
        "lang": "vi",
        "device": "cpu",
        # "ocr_version": "PP-OCRv5",
        "ocr_version": "PP-OCRv4",
        "use_textline_orientation": False,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "det_model_dir": None,
        "rec_model_dir": None
    },
    "input": {
        "empty_image_policy": "raise",
        "color_mode": "rgb",
        "gray_to_rgb_code": "COLOR_GRAY2RGB",
        "bgr_to_rgb_code": "COLOR_BGR2RGB",
        "bgra_to_rgb_code": "COLOR_BGRA2RGB",
    },
    "parse": {
        "result_root_key": "res",
        "texts_key": "rec_texts",
        "scores_key": "rec_scores",
        "polys_key": "dt_polys",
        "skip_empty_text": True,
        "strip_text": True,
        "polygon_cast_int": True,
        "fallback_empty_polygon": True,
        "confidence_scale": 1.0,
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


@dataclass
class PaddleParams:
    # lang: str = "en"
    lang: str = "vi"
    device: str = "cpu"
    # ocr_version: str = "PP-OCRv5"
    ocr_version: str = "PP-OCRv4"

    use_textline_orientation: bool = False
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False

    det_model_dir: str | None = None
    rec_model_dir: str | None = None


class PaddleModel:
    def __init__(
        self,
        params: PaddleParams | None = None,
        engine_config: dict | None = None,
    ):
        if params is None and engine_config is not None:
            params = PaddleParams(**engine_config)

        self.params = params or PaddleParams()

        kwargs = {
            k: v
            for k, v in asdict(self.params).items()
            if v is not None
        }

        self._engine = PaddleOCR(**kwargs)


class Paddle(BaseOCR):
    def __init__(
        self,
        model: PaddleModel | None = None,
        config: dict | None = None,
    ):
        super().__init__()

        self.config = _deep_merge(DEFAULT_PADDLE_CONFIG, config or {})
        self._validate_config()

        engine_cfg = self.config["engine"]
        self.model = model or PaddleModel(engine_config=engine_cfg)

    def _validate_config(self) -> None:
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

        if parse_cfg["confidence_scale"] <= 0:
            raise ValueError("parse.confidence_scale must be > 0")

        required_parse_keys = {
            "result_root_key",
            "texts_key",
            "scores_key",
            "polys_key",
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
            self._warn(f"WARN [PaddleOCR] {exc}")

        if self.config["runtime"]["error_policy"] == "raise":
            raise

        return OCRResult(texts=[])

    def _resolve_cv2_constant(self, name: str) -> int:
        try:
            return getattr(cv2, name)
        except AttributeError as e:
            raise ValueError(f"Unknown cv2 constant: {name}") from e

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        cfg = self.config["input"]

        if image is None or getattr(image, "size", 0) == 0:
            return image

        if image.ndim == 2:
            return cv2.cvtColor(
                image,
                self._resolve_cv2_constant(cfg["gray_to_rgb_code"]),
            )

        if image.ndim == 3:
            if image.shape[2] == 3:
                return cv2.cvtColor(
                    image,
                    self._resolve_cv2_constant(cfg["bgr_to_rgb_code"]),
                )
            if image.shape[2] == 4:
                return cv2.cvtColor(
                    image,
                    self._resolve_cv2_constant(cfg["bgra_to_rgb_code"]),
                )

        return image

    # def _parse_result(self, raw: Any) -> OCRResult:
    #     parse_cfg = self.config["parse"]

    #     if not raw:
    #         return OCRResult(texts=[])

    #     blocks: list[TextBlock] = []

    #     for res in raw:
    #         data = getattr(res, "json", {}).get(
    #             parse_cfg["result_root_key"], {}
    #         )

    #         rec_texts = data.get(parse_cfg["texts_key"], []) or []
    #         rec_scores = data.get(parse_cfg["scores_key"], []) or []
    #         dt_polys = data.get(parse_cfg["polys_key"], []) or []

    #         if (
    #             parse_cfg.get("fallback_empty_polygon", True)
    #             and not dt_polys
    #             and rec_texts
    #         ):
    #             dt_polys = [[] for _ in rec_texts]

    #         for text, conf, poly in zip(rec_texts, rec_scores, dt_polys):
    #             normalized_text = (
    #                 text.strip()
    #                 if parse_cfg.get("strip_text", True) and isinstance(text, str)
    #                 else text
    #             )

    #             if (
    #                 parse_cfg.get("skip_empty_text", True)
    #                 and (not normalized_text or str(normalized_text).strip() == "")
    #             ):
    #                 continue

    #             if parse_cfg["polygon_cast_int"] and poly:
    #                 polygon = [
    #                     (int(p[0]), int(p[1])) for p in poly
    #                 ]
    #             else:
    #                 polygon = [
    #                     (p[0], p[1]) for p in poly
    #                 ] if poly else []

    #             confidence = float(conf) * float(
    #                 parse_cfg["confidence_scale"]
    #             )

    #             block = TextBlock(
    #                 text=str(normalized_text),
    #                 bounding_polygon=polygon,
    #                 confidence=confidence,
    #             )
    #             blocks.append(block)

    #     return OCRResult(texts=blocks)

    def _parse_result(self, raw: Any) -> OCRResult:
        if not raw:
            return OCRResult(texts=[])

        blocks: list[TextBlock] = []

        try:
            lines = raw[0]

            for line in lines:
                if not line or len(line) < 2:
                    continue

                polygon = line[0]
                text = line[1][0]
                confidence = float(line[1][1])

                if not text or str(text).strip() == "":
                    continue

                polygon = [
                    (int(point[0]), int(point[1]))
                    for point in polygon
                ]

                blocks.append(
                    TextBlock(
                        text=str(text).strip(),
                        bounding_polygon=polygon,
                        confidence=confidence,
                    )
                )

        except Exception as e:
            self._warn(f"Parse OCR result failed: {e}")
            return OCRResult(texts=[])

        return OCRResult(texts=blocks)

    def _recognize_single(self, image: np.ndarray) -> OCRResult:
        if image is None or getattr(image, "size", 0) == 0:
            return self._handle_empty_image()

        try:
            rgb = self._to_rgb(image)
            # raw = self.model._engine.predict(rgb)
            raw = self.model._engine.ocr(rgb, cls=True)
            return self._parse_result(raw)
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
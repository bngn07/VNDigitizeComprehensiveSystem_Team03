from __future__     import annotations

from collections    import deque
from dataclasses    import dataclass, field
from typing         import Callable, Any

import cv2
import numpy as np

from .code          import CodeDetector
from .decide        import DecisionEngine, DecisionResult, LABEL_SKIP, LABEL_CLEAN
from .rotate        import RotationDetector
from .geometry      import detect_document, four_point_transform


DEFAULT_PREPROCESS_CONFIG: dict = {
    "recipe_separator": ",",
    "unknown_step_policy": "skip",
    "warn_on_unknown_step": True,
    "warn_on_perspective_skip": True,
    "empty_image_policy": "raise",
    "copy_input_image": True,
    "status_map": {
        "skip": "skipped_blank",
        "clean": "clean",
        "heavy": "heavy",
    },
    "metadata_keys": {
        "status": "status",
        "qr_codes": "qr_codes",
    },
    "grayscale": {
        "enabled": True,
        "conversion_code": "COLOR_BGR2GRAY",
    },
    "gaussian_blur": {
        "enabled": True,
        "kernel_size": [5, 5],
        "sigma_x": 0,
    },
    "deskew": {
        "enabled": True,
        "threshold": {
            "max_value": 255,
            "mode": "THRESH_BINARY|THRESH_OTSU",
        },
        "angle_switch_threshold": -45,
        "max_angle": 45,
        "rotation_scale": 1.0,
        "interpolation": "INTER_CUBIC",
        "border_mode": "BORDER_REPLICATE",
        "border_value": 0,
        "min_foreground_pixels": 1,
    },
    "autocrop": {
        "enabled": True,
        "threshold": {
            "max_value": 255,
            "mode": "THRESH_BINARY|THRESH_OTSU",
        },
        "min_area_ratio": 0.1,
    },
    "adaptive_threshold": {
        "enabled": True,
        "max_value": 255,
        "method": "ADAPTIVE_THRESH_GAUSSIAN_C",
        "threshold_type": "THRESH_BINARY",
        "block_size": 11,
        "c": 2,
    },
    "sharpening": {
        "enable": True,
        "kernel": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "ddepth": -1,
    },
    "perspective": {
        "enabled": True,
        "resize_height": 500,
        "resize_interpolation": "INTER_LINEAR",
        "on_document_not_found": "return_original",
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
class PreprocessResult:
    image: np.ndarray
    metadata: dict
    decision: DecisionResult | None = field(default=None, repr=False)

    def __str__(self) -> str:
        parts = []

        parts.append("IMAGE")
        if self.image is not None:
            h, w = self.image.shape[:2]
            c = self.image.shape[2] if self.image.ndim > 2 else 1
            parts.append(f" size : {w} x {h}")
            parts.append(f" channels : {c}")
            parts.append(f" dtype : {self.image.dtype}")
        else:
            parts.append(" None")

        parts.append("\nDECISION")
        if self.decision:
            parts.append(f" label : {self.decision.label_name}")
            parts.append(f" confidence : {self.decision.confidence:.4f}")
            parts.append(f" recipe : {self.decision.recipe}")
            parts.append(
                f" probs : "
                f"{ {k: f'{v:.3f}' for k, v in self.decision.probs.items()} }"
            )
        else:
            parts.append(" no decision (engine not loaded)")

        metadata_status_key = "status"
        metadata_qr_key = "qr_codes"

        parts.append("\nMETADATA")
        parts.append(f" status : {self.metadata.get(metadata_status_key)}")
        parts.append(f" qr_count : {len(self.metadata.get(metadata_qr_key, []))}")

        qr_codes = self.metadata.get(metadata_qr_key, [])
        if qr_codes:
            parts.append("\nQR OBJECTS")
            for i, qr in enumerate(qr_codes, 1):
                parts.append(f" [{i}]")

                if hasattr(qr, "__dict__"):
                    data = qr.__dict__
                elif hasattr(qr, "__slots__"):
                    data = {s: getattr(qr, s) for s in qr.__slots__}
                elif isinstance(qr, dict):
                    data = qr
                else:
                    data = {"value": str(qr)}

                for k, v in data.items():
                    parts.append(f" {k}: {v}")

        return "\n".join(parts)


class Preprocessing:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.cv_cfg = _deep_merge(
            DEFAULT_PREPROCESS_CONFIG,
            self.config.get("image_preprocessor", {}),
        )

        decide_cfg = dict(self.config.get("decide_engine", {}))
        code_cfg = dict(self.config.get("code_detector", {}))
        osd_cfg = dict(self.config.get("tesseract", {}))
        self.geo_cfg = dict(self.config.get("geometry", {}))

        if "provider" not in decide_cfg:
            decide_cfg["provider"] = decide_cfg.pop("model_path", None)

        self.engine = DecisionEngine(**decide_cfg)
        self.code_detector = CodeDetector(**code_cfg)
        self.rotation_detector = RotationDetector(**osd_cfg)

        self._qr_buffer: list = []

        self.step_menu: dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "grayscale": self._to_grayscale,
            "orientation": self._orient,
            "perspective": self._perspective_correct,
            "denoise": self._denoise,
            "deskew": self._deskew,
            "autocrop": self._autocrop,
            "adaptive_threshold": self._adaptive_threshold,
            "sharpen": self._sharpen,
            "qr_detect": self._qr_detect,
        }

        self._validate_config()

    def _validate_config(self) -> None:
        if self.cv_cfg["unknown_step_policy"] not in {"skip", "raise"}:
            raise ValueError("unknown_step_policy must be 'skip' or 'raise'")

        if self.cv_cfg["empty_image_policy"] not in {"raise", "return_original"}:
            raise ValueError("empty_image_policy must be 'raise' or 'return_original'")

        if self.cv_cfg["perspective"]["on_document_not_found"] not in {
            "return_original",
            "raise",
        }:
            raise ValueError(
                "perspective.on_document_not_found must be 'return_original' or 'raise'"
            )

        block_size = self.cv_cfg["adaptive_threshold"]["block_size"]
        if block_size <= 1 or block_size % 2 == 0:
            raise ValueError(
                "adaptive_threshold.block_size must be an odd integer > 1"
            )

        min_area_ratio = self.cv_cfg["autocrop"]["min_area_ratio"]
        if not 0 <= min_area_ratio <= 1:
            raise ValueError("autocrop.min_area_ratio must be in [0, 1]")

        if self.cv_cfg["perspective"]["resize_height"] <= 0:
            raise ValueError("perspective.resize_height must be > 0")

    @staticmethod
    def _resolve_cv2_constant(name: str) -> int:
        try:
            return getattr(cv2, name)
        except AttributeError as e:
            raise ValueError(f"Unknown cv2 constant: {name}") from e

    def _parse_cv2_flags(self, value: str | int) -> int:
        if isinstance(value, int):
            return value

        parts = [p.strip() for p in str(value).split("|")]
        flags = 0
        for part in parts:
            flags |= self._resolve_cv2_constant(part)
        return flags

    def _warn(self, message: str) -> None:
        print(message)

    def _handle_empty_image(self, image: np.ndarray | None) -> np.ndarray:
        if self.cv_cfg["empty_image_policy"] == "raise":
            raise ValueError("Input image is empty.")
        return image

    def _build_metadata(self, status: str, qr_list: list) -> dict:
        keys = self.cv_cfg["metadata_keys"]
        return {
            keys["status"]: status,
            keys["qr_codes"]: qr_list,
        }

    def _split_recipe(self, recipe: str | None) -> deque[str]:
        if not recipe:
            return deque()

        separator = self.cv_cfg["recipe_separator"]
        return deque(
            step.strip().lower()
            for step in recipe.split(separator)
            if step.strip()
        )

    def _status_from_decision(self, decision: DecisionResult) -> str:
        status_map = self.cv_cfg["status_map"]
        if decision.label == LABEL_SKIP:
            return status_map["skip"]
        if decision.label == LABEL_CLEAN:
            return status_map["clean"]
        return status_map["heavy"]

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        cfg = self.cv_cfg["grayscale"]
        if not cfg.get("enabled", True):
            return image

        if image.ndim == 3:
            return cv2.cvtColor(
                image,
                self._resolve_cv2_constant(cfg["conversion_code"]),
            )
        return image.copy()

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        cfg = self.cv_cfg["gaussian_blur"]
        if not cfg.get("enabled", True):
            return image

        return cv2.GaussianBlur(
            image,
            tuple(cfg["kernel_size"]),
            cfg["sigma_x"],
        )

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        cfg = self.cv_cfg["deskew"]
        if not cfg.get("enabled", True):
            return gray

        thresh = cv2.threshold(
            cv2.bitwise_not(gray),
            0,
            cfg["threshold"]["max_value"],
            self._parse_cv2_flags(cfg["threshold"]["mode"]),
        )[1]

        coords = np.column_stack(np.where(thresh > 0))
        if coords.size < cfg.get("min_foreground_pixels", 1):
            return gray

        angle = cv2.minAreaRect(coords)[-1]
        switch_threshold = cfg["angle_switch_threshold"]
        angle = -(90 + angle) if angle < switch_threshold else -angle

        if abs(angle) > cfg["max_angle"]:
            return gray

        h, w = gray.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (w // 2, h // 2),
            angle,
            cfg["rotation_scale"],
        )

        return cv2.warpAffine(
            gray,
            matrix,
            (w, h),
            flags=self._resolve_cv2_constant(cfg["interpolation"]),
            borderMode=self._resolve_cv2_constant(cfg["border_mode"]),
            borderValue=cfg.get("border_value", 0),
        )

    def _autocrop(self, gray: np.ndarray) -> np.ndarray:
        cfg = self.cv_cfg["autocrop"]
        if not cfg.get("enabled", True):
            return gray

        thresh = cv2.threshold(
            cv2.bitwise_not(gray),
            0,
            cfg["threshold"]["max_value"],
            self._parse_cv2_flags(cfg["threshold"]["mode"]),
        )[1]

        coords = cv2.findNonZero(thresh)
        if coords is None:
            return gray

        x, y, w, h = cv2.boundingRect(coords)
        if w * h < cfg["min_area_ratio"] * gray.shape[0] * gray.shape[1]:
            return gray

        return gray[y : y + h, x : x + w]

    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        cfg = self.cv_cfg["adaptive_threshold"]
        if not cfg.get("enabled", True):
            return image

        return cv2.adaptiveThreshold(
            image,
            cfg["max_value"],
            self._resolve_cv2_constant(cfg["method"]),
            self._resolve_cv2_constant(cfg["threshold_type"]),
            cfg["block_size"],
            cfg["c"],
        )

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        cfg = self.cv_cfg["sharpening"]
        if not cfg.get("enable", True):
            return image

        kernel = np.array(cfg["kernel"])
        return cv2.filter2D(image, cfg.get("ddepth", -1), kernel)

    def _perspective_correct(self, image: np.ndarray) -> np.ndarray:
        cfg = self.cv_cfg["perspective"]
        if not cfg.get("enabled", True):
            return image

        resize_height = cfg["resize_height"]
        ratio = image.shape[0] / float(resize_height)

        image_small = cv2.resize(
            image,
            (int(image.shape[1] / ratio), resize_height),
            interpolation=self._resolve_cv2_constant(cfg["resize_interpolation"]),
        )

        doc_cnt = detect_document(image_small, config=self.geo_cfg)

        if doc_cnt is None:
            if self.cv_cfg.get("warn_on_perspective_skip", True):
                self._warn(
                    "WARN [perspective_correct] no document contour found -> skipping."
                )

            if cfg["on_document_not_found"] == "raise":
                raise ValueError("No document contour found for perspective correction.")

            return image

        doc_cnt = doc_cnt.reshape(4, 2) * ratio
        return four_point_transform(image, doc_cnt, config=self.geo_cfg)

    def _orient(self, image: np.ndarray) -> np.ndarray:
        return self.rotation_detector.correct(image)

    def _qr_detect(self, image: np.ndarray) -> np.ndarray:
        self._qr_buffer.extend(self.code_detector.detect(image))
        return image

    def _run_step(self, step: str, current: np.ndarray) -> np.ndarray:
        func = self.step_menu.get(step)
        if func is not None:
            return func(current)

        message = f"WARN [Preprocessing] unknown step '{step}' -> skipping."
        if self.cv_cfg["unknown_step_policy"] == "raise":
            raise ValueError(message)

        if self.cv_cfg.get("warn_on_unknown_step", True):
            self._warn(message)

        return current

    def process(self, image: np.ndarray) -> PreprocessResult:
        self._qr_buffer = []

        if image is None or getattr(image, "size", 0) == 0:
            handled = self._handle_empty_image(image)
            return PreprocessResult(
                image    = handled,
                metadata = self._build_metadata(
                    self.cv_cfg["status_map"]["skip"],
                    [],
                ),
                decision = None,
            )

        decision = self.engine.evaluate(image)

        if decision.label == LABEL_SKIP:
            return PreprocessResult(
                image=image,
                metadata=self._build_metadata(
                    self._status_from_decision(decision),
                    [],
                ),
                decision=decision,
            )

        current = image.copy() if self.cv_cfg.get("copy_input_image", True) else image
        step_queue = self._split_recipe(decision.recipe)

        while step_queue:
            step = step_queue.popleft()
            current = self._run_step(step, current)

        return PreprocessResult(
            image    = current,
            metadata = self._build_metadata(
                self._status_from_decision(decision),
                self._qr_buffer,
            ),
            decision =decision,
        )
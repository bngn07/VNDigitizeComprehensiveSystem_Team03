from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .code import CodeDetector
from .decide import (
    DecisionEngine,
    DecideML,
    DecisionResult,
    LABEL_SKIP,
    LABEL_CLEAN,
)
from .rotate import RotationDetector
from .geometry import detect_document, four_point_transform


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
            parts.append(f"  size     : {w} x {h}")
            parts.append(f"  channels : {c}")
            parts.append(f"  dtype    : {self.image.dtype}")
        else:
            parts.append("  None")

        parts.append("\nDECISION")
        if self.decision:
            parts.append(f"  label      : {self.decision.label_name}")
            parts.append(f"  confidence : {self.decision.confidence:.4f}")
            parts.append(f"  recipe     : {self.decision.recipe}")
            parts.append(
                f"  probs      : "
                f"{ {k: f'{v:.3f}' for k, v in self.decision.probs.items()} }"
            )
        else:
            parts.append("  no decision (engine not loaded)")

        parts.append("\nMETADATA")
        parts.append(f"  status   : {self.metadata.get('status')}")
        parts.append(f"  qr_count : {len(self.metadata.get('qr_codes', []))}")

        qr_codes = self.metadata.get("qr_codes", [])
        if qr_codes:
            parts.append("\nQR OBJECTS")
            for i, qr in enumerate(qr_codes, 1):
                parts.append(f"  [{i}]")

                if hasattr(qr, "__dict__"):
                    data = qr.__dict__
                elif hasattr(qr, "__slots__"):
                    data = {s: getattr(qr, s) for s in qr.__slots__}
                elif isinstance(qr, dict):
                    data = qr
                else:
                    data = {"value": str(qr)}

                for k, v in data.items():
                    parts.append(f"    {k}: {v}")

        return "\n".join(parts)


class Preprocessing:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.cv_cfg = self.config.get("image_preprocessor", {})
    
        decide_cfg = dict(self.config.get("decide_engine", {}))
        code_cfg = self.config.get("code_detector", {})
        osd_cfg = self.config.get("tesseract", {})

        if "provider" not in decide_cfg:
            decide_cfg["provider"] = decide_cfg.pop("model_path", "")

        self.engine = DecisionEngine(**decide_cfg)
        self.code_detector = CodeDetector(**code_cfg)
        self.rotation_detector = RotationDetector(**osd_cfg)

        self._qr_buffer: list = []

        self.step_menu: dict[str, callable] = {
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

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        gb_cfg = self.cv_cfg.get("gaussian_blur", {})
        k_size = tuple(gb_cfg.get("kernel_size", (5, 5)))
        sigma = gb_cfg.get("sigma_x", 0)

        return cv2.GaussianBlur(gray, k_size, sigma)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        if not self.cv_cfg.get("enable_deskew", True):
            return gray

        max_angle = self.cv_cfg.get("deskew", {}).get("max_angle", 45)

        thresh = cv2.threshold(
            cv2.bitwise_not(gray),
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )[1]

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle

        if abs(angle) > max_angle:
            return gray

        h, w = gray.shape[:2]

        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

        return cv2.warpAffine(
            gray,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _autocrop(self, gray: np.ndarray) -> np.ndarray:
        if not self.cv_cfg.get("enable_autocrop", True):
            return gray

        thresh = cv2.threshold(
            cv2.bitwise_not(gray),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]

        coords = cv2.findNonZero(thresh)

        if coords is None:
            return gray

        x, y, w, h = cv2.boundingRect(coords)

        if w * h < 0.1 * gray.shape[0] * gray.shape[1]:
            return gray

        return gray[y:y + h, x:x + w]

    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        cfg = self.cv_cfg.get(
            "adaptive_threshold",
            {
                "max_value": 255,
                "block_size": 11,
                "c": 2,
            },
        )

        return cv2.adaptiveThreshold(
            image,
            cfg["max_value"],
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            cfg["block_size"],
            cfg["c"],
        )

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        sharp_cfg = self.cv_cfg.get("sharpening", {})

        if not sharp_cfg.get("enable", True):
            return image

        kernel = np.array(
            sharp_cfg.get(
                "kernel",
                [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
            )
        )

        return cv2.filter2D(image, -1, kernel)

    def _perspective_correct(self, image: np.ndarray) -> np.ndarray:
        ratio = image.shape[0] / 500.0

        image_small = cv2.resize(
            image,
            (int(image.shape[1] / ratio), 500),
        )

        doc_cnt = detect_document(image_small)

        if doc_cnt is None:
            print(
                "WARN [perspective_correct] "
                "no document contour found → skipping."
            )
            return image

        doc_cnt = doc_cnt.reshape(4, 2) * ratio

        return four_point_transform(image, doc_cnt)

    def _orient(self, image: np.ndarray) -> np.ndarray:
        return self.rotation_detector.correct(image)

    def _qr_detect(self, image: np.ndarray) -> np.ndarray:
        self._qr_buffer.extend(self.code_detector.detect(image))
        return image

    @staticmethod
    def _build_metadata(status: str, qr_list: list) -> dict:
        return {
            "status": status,
            "qr_codes": qr_list,
        }

    def process(self, image: np.ndarray) -> PreprocessResult:
        self._qr_buffer = []

        decision = self.engine.evaluate(image)

        if decision.label == LABEL_SKIP:
            return PreprocessResult(
                image=image,
                metadata=self._build_metadata("skipped_blank", []),
                decision=decision,
            )

        current = image.copy()

        step_queue = deque(
            step.strip().lower()
            for step in decision.recipe.split(",")
        )

        while step_queue:
            step = step_queue.popleft()

            if step in self.step_menu:
                current = self.step_menu[step](current)
            else:
                print(
                    f"WARN [Preprocessing] "
                    f"unknown step '{step}' — skipping."
                )

        status = (
            "clean"
            if decision.label == LABEL_CLEAN
            else "heavy"
        )

        return PreprocessResult(
            image=current,
            metadata=self._build_metadata(status, self._qr_buffer),
            decision=decision,
        )
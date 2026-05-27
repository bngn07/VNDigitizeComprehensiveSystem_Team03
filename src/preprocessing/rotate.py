from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


DEFAULT_ROTATION_CONFIG: dict = {
    "config": "--psm 0",
    "lang": None,
    "nice": 0,
    "timeout": 0,
    "min_confidence": 0.1,
    "normalize_confidence": True,
    "confidence_scale": 100.0,
    "unknown_script_value": "Unknown",
    "empty_image_policy": "return_default",
    "error_policy": "return_default",
    "warn_on_error": True,
    "verbose": False,
    "apply_rotation": True,
    "skip_if_angle_zero": True,
    "supported_angles": [0, 90, 180, 270],
    "angle_to_rotation_map": {
        90: "ROTATE_90_CLOCKWISE",
        180: "ROTATE_180",
        270: "ROTATE_90_COUNTERCLOCKWISE",
    },
    "on_error_angle": 0.0,
    "on_error_confidence": 0.0,
    "on_error_script": "Unknown",
}


@dataclass
class RotationResult:
    angle: float
    confidence: float
    script: str = "Unknown"


class RotationDetector:
    def __init__(self, **config: Any):
        self.config = {
            **DEFAULT_ROTATION_CONFIG,
            **config,
        }
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config["empty_image_policy"] not in {"return_default", "raise"}:
            raise ValueError(
                "empty_image_policy must be 'return_default' or 'raise'"
            )

        if self.config["error_policy"] not in {"return_default", "raise"}:
            raise ValueError(
                "error_policy must be 'return_default' or 'raise'"
            )

        if self.config["confidence_scale"] <= 0:
            raise ValueError("confidence_scale must be > 0")

        if self.config["min_confidence"] < 0:
            raise ValueError("min_confidence must be >= 0")

        if not isinstance(self.config["supported_angles"], list):
            raise ValueError("supported_angles must be a list")

        if not isinstance(self.config["angle_to_rotation_map"], dict):
            raise ValueError("angle_to_rotation_map must be a dict")

    def _default_result(self) -> RotationResult:
        return RotationResult(
            angle=float(self.config["on_error_angle"]),
            confidence=float(self.config["on_error_confidence"]),
            script=str(self.config["on_error_script"]),
        )

    def _handle_empty_image(self) -> RotationResult:
        if self.config["empty_image_policy"] == "raise":
            raise ValueError("Input image is empty.")
        return self._default_result()

    def _normalize_confidence(self, confidence: float) -> float:
        if self.config["normalize_confidence"]:
            return confidence / float(self.config["confidence_scale"])
        return confidence

    def detect(self, image: np.ndarray) -> RotationResult:
        if image is None or image.size == 0:
            return self._handle_empty_image()

        try:
            osd_data = pytesseract.image_to_osd(
                image,
                lang=self.config["lang"],
                config=self.config["config"],
                nice=self.config["nice"],
                output_type=Output.DICT,
                timeout=self.config["timeout"],
            )

            angle = float(osd_data.get("rotate", 0.0))
            raw_confidence = float(osd_data.get("orientation_conf", 0.0))
            confidence = self._normalize_confidence(raw_confidence)
            script = str(
                osd_data.get("script", self.config["unknown_script_value"])
            )

            result = RotationResult(
                angle=angle,
                confidence=confidence,
                script=script,
            )

        except Exception as e:
            if self.config["warn_on_error"]:
                print(f"Error during rotation detection: {e}")

            if self.config["error_policy"] == "raise":
                raise

            result = self._default_result()

        if self.config["verbose"]:
            print(
                f"Ran rotation detection: "
                f"{result.angle} degrees, confidence: {result.confidence}"
            )

        return result

    def correct(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            if self.config["empty_image_policy"] == "raise":
                raise ValueError("Input image is empty.")
            return image

        result = self.detect(image)

        if result.confidence < self.config["min_confidence"]:
            return image

        if self.config["skip_if_angle_zero"] and result.angle == 0:
            return image

        if not self.config["apply_rotation"]:
            return image

        angle = int(result.angle)

        if angle not in self.config["supported_angles"]:
            return image

        rotate_name = self.config["angle_to_rotation_map"].get(angle)
        if not rotate_name:
            return image

        rotate_code = getattr(cv2, rotate_name, None)
        if rotate_code is None:
            raise ValueError(f"Unknown cv2 rotate code: {rotate_name}")

        return cv2.rotate(image, rotate_code)
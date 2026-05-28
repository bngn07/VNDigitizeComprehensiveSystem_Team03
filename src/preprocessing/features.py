from __future__ import annotations

import cv2
import numpy as np

from constants  import (COEFF_EPSILON, 
                        RESIZE_HEIGHT, 
                        RESIZE_WIDTH, 
                        WHITE_RATIO_TOLERANCE)


def extract_features(image: np.ndarray) -> dict[str, float]:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    gray = image
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)

    std_val = float(np.std(gray))
    mean_intensity = float(np.mean(gray))
    background = float(np.median(gray))

    white_ratio = float(
        np.sum(np.abs(gray.astype(np.int32) - int(background)) <= WHITE_RATIO_TOLERANCE)
        / gray.size
    )
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edge_density = float(np.count_nonzero(cv2.Canny(gray, 100, 200)) / gray.size)
    coeff_variation = std_val / (mean_intensity + COEFF_EPSILON)

    return {
        "white_ratio": white_ratio,
        "std_val": std_val,
        "coeff_variation": coeff_variation,
        "laplacian_var": laplacian_var,
        "mean_intensity": mean_intensity,
        "edge_density": edge_density,
    }


def to_vector(features: dict[str, float], feature_keys: list[str]) -> np.ndarray:
    return np.array([features[key] for key in feature_keys], dtype=np.float32)

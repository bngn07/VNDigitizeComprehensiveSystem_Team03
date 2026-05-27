from __future__ import annotations

import cv2
import numpy as np


DEFAULT_GEOMETRY_CONFIG: dict = {
    "grayscale_conversion_code": "COLOR_BGR2GRAY",
    "document_detection": {
        "blur_kernel_size": [3, 3],
        "blur_sigma_x": 0,
        "canny_threshold1": 30,
        "canny_threshold2": 100,
        "morph_kernel_size": [5, 5],
        "morph_kernel_dtype": "uint8",
        "dilate_iterations": 1,
        "erode_iterations": 1,
        "contour_retrieval_mode": "RETR_LIST",
        "contour_approx_method": "CHAIN_APPROX_SIMPLE",
        "top_k_contours": 10,
        "polygon_epsilon_ratio": 0.02,
        "polygon_closed": True,
        "required_polygon_vertices": 4,
        "sort_contours_desc": True
    },
    "perspective_transform": {
        "output_dtype": "float32",
        "interpolation": "INTER_LINEAR",
        "border_mode": "BORDER_CONSTANT",
        "border_value": 0
    }
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


def _resolve_cv2_constant(name: str) -> int:
    try:
        return getattr(cv2, name)
    except AttributeError as e:
        raise ValueError(f"Unknown cv2 constant: {name}") from e


def _resolve_numpy_dtype(name: str):
    try:
        return getattr(np, name)
    except AttributeError as e:
        raise ValueError(f"Unknown numpy dtype: {name}") from e


def _validate_config(config: dict) -> None:
    doc_cfg = config["document_detection"]
    persp_cfg = config["perspective_transform"]

    blur_kernel = doc_cfg["blur_kernel_size"]
    morph_kernel = doc_cfg["morph_kernel_size"]

    if len(blur_kernel) != 2 or any(v <= 0 for v in blur_kernel):
        raise ValueError("document_detection.blur_kernel_size must have 2 positive integers")

    if len(morph_kernel) != 2 or any(v <= 0 for v in morph_kernel):
        raise ValueError("document_detection.morph_kernel_size must have 2 positive integers")

    if doc_cfg["blur_sigma_x"] < 0:
        raise ValueError("document_detection.blur_sigma_x must be >= 0")

    if doc_cfg["top_k_contours"] <= 0:
        raise ValueError("document_detection.top_k_contours must be > 0")

    if doc_cfg["polygon_epsilon_ratio"] <= 0:
        raise ValueError("document_detection.polygon_epsilon_ratio must be > 0")

    if doc_cfg["required_polygon_vertices"] <= 0:
        raise ValueError("document_detection.required_polygon_vertices must be > 0")

    if doc_cfg["dilate_iterations"] < 0 or doc_cfg["erode_iterations"] < 0:
        raise ValueError("morph iterations must be >= 0")

    _resolve_cv2_constant(config["grayscale_conversion_code"])
    _resolve_cv2_constant(doc_cfg["contour_retrieval_mode"])
    _resolve_cv2_constant(doc_cfg["contour_approx_method"])
    _resolve_cv2_constant(persp_cfg["interpolation"])
    _resolve_cv2_constant(persp_cfg["border_mode"])
    _resolve_numpy_dtype(doc_cfg["morph_kernel_dtype"])
    _resolve_numpy_dtype(persp_cfg["output_dtype"])


def _get_config(config: dict | None = None) -> dict:
    merged = _deep_merge(DEFAULT_GEOMETRY_CONFIG, config or {})
    _validate_config(merged)
    return merged


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(
    image: np.ndarray,
    pts: np.ndarray,
    config: dict | None = None,
) -> np.ndarray:
    cfg = _get_config(config)
    persp_cfg = cfg["perspective_transform"]

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=persp_cfg["output_dtype"],
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=_resolve_cv2_constant(persp_cfg["interpolation"]),
        borderMode=_resolve_cv2_constant(persp_cfg["border_mode"]),
        borderValue=persp_cfg["border_value"],
    )

    return warped


def detect_document(
    image: np.ndarray,
    config: dict | None = None,
) -> np.ndarray | None:
    cfg = _get_config(config)
    doc_cfg = cfg["document_detection"]

    if image is None or image.size == 0:
        return None

    if image.ndim == 3 and image.shape[2] in (3, 4):
        gray = cv2.cvtColor(
            image,
            _resolve_cv2_constant(cfg["grayscale_conversion_code"]),
        )
    else:
        gray = image

    blurred = cv2.GaussianBlur(
        gray,
        tuple(doc_cfg["blur_kernel_size"]),
        doc_cfg["blur_sigma_x"],
    )

    edged = cv2.Canny(
        blurred,
        doc_cfg["canny_threshold1"],
        doc_cfg["canny_threshold2"],
    )

    kernel = np.ones(
        tuple(doc_cfg["morph_kernel_size"]),
        dtype=_resolve_numpy_dtype(doc_cfg["morph_kernel_dtype"]),
    )

    edged = cv2.dilate(edged, kernel, iterations=doc_cfg["dilate_iterations"])
    edged = cv2.erode(edged, kernel, iterations=doc_cfg["erode_iterations"])

    contours, _ = cv2.findContours(
        edged,
        _resolve_cv2_constant(doc_cfg["contour_retrieval_mode"]),
        _resolve_cv2_constant(doc_cfg["contour_approx_method"]),
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=doc_cfg["sort_contours_desc"],
    )

    for contour in contours[: doc_cfg["top_k_contours"]]:
        perimeter = cv2.arcLength(contour, doc_cfg["polygon_closed"])
        epsilon = doc_cfg["polygon_epsilon_ratio"] * perimeter
        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            doc_cfg["polygon_closed"],
        )

        if len(approx) == doc_cfg["required_polygon_vertices"]:
            return approx

    return None
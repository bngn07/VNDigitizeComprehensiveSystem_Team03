from __future__ import annotations

import cv2
import numpy as np

from .code_detector import CodeDetector
from .geometry import detect_document, four_point_transform
from .rotate import RotationDetector


rotation_detector = RotationDetector()
code_detector = CodeDetector()


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def orient(image: np.ndarray) -> np.ndarray:
    return rotation_detector.correct(image)


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 0)


def deskew(gray: np.ndarray) -> np.ndarray:
    thresh = cv2.threshold(cv2.bitwise_not(gray), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size < 1:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) > 45:
        return gray

    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def autocrop(gray: np.ndarray) -> np.ndarray:
    thresh = cv2.threshold(cv2.bitwise_not(gray), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return gray

    x, y, width, height = cv2.boundingRect(coords)
    if width * height < 0.1 * gray.shape[0] * gray.shape[1]:
        return gray

    return gray[y : y + height, x : x + width]


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def perspective_correct(image: np.ndarray) -> np.ndarray:
    resize_height = 500
    ratio = image.shape[0] / float(resize_height)
    resize_width = int(image.shape[1] / ratio)
    small = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)

    contour = detect_document(small)
    if contour is None:
        return image

    contour = contour.reshape(4, 2) * ratio
    return four_point_transform(image, contour)


def qr_detect(image: np.ndarray) -> tuple[np.ndarray, list]:
    results = code_detector.detect(image)
    return image, results
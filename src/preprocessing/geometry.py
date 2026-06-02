from __future__ import annotations

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    points  = points.reshape(4, 2)
    rect    = np.zeros((4, 2), dtype = np.float32)

    sums    = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]

    diffs   = np.diff(points, axis = 1)
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]

    return rect


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect

    width_a     = np.linalg.norm(bottom_right - bottom_left)
    width_b     = np.linalg.norm(top_right - top_left)
    max_width   = int(max(width_a, width_b))

    height_a    = np.linalg.norm(top_right - bottom_right)
    height_b    = np.linalg.norm(top_left - bottom_left)
    max_height  = int(max(height_a, height_b))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def detect_document(image: np.ndarray) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None

    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged   = cv2.Canny(blurred, 30, 100)

    kernel  = np.ones((5, 5), dtype=np.uint8)
    edged   = cv2.dilate(edged, kernel, iterations = 1)
    edged   = cv2.erode(edged, kernel, iterations = 1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours    = sorted(contours, key = cv2.contourArea, reverse = True)

    for contour in contours[:10]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx

    return None
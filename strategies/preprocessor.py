import cv2
import numpy as np
from typing import Tuple
from .base import IBlankDetector
import joblib

class FastBlankDetector(IBlankDetector):
    def __init__(
        self, 
        gray_threshold_value: float = 127,
        upper_threshold: float = 0.95,
        lower_threshold: float = 0.01
    ):
        self.gray_threshold_value = gray_threshold_value
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
    
    def is_blank(
        self,
        image: np.ndarray
    ) -> Tuple[bool, float, str]:

        black_ratio = np.sum(image < self.gray_threshold_value) / image.size

        is_blank = black_ratio < self.lower_threshold or black_ratio > self.upper_threshold
    
        return (is_blank, black_ratio, "black-ratio certain")

class RFBlankDetector(IBlankDetector):
    def __init__(self, 
        model_path: str, 
        confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Không thể load mô hình RF tại {model_path}. Chi tiết lỗi: {e}")

    def _extract_features(self, image: np.ndarray) -> tuple:
        '''
            Black page shouldnt be inputed and presented here.
        '''
        black_threshold = 127

        # Black pixel ratio
        black_ratio = np.sum(image < black_threshold) / image.size

        # std dev
        std = np.std(image)

        # hist
        hist = np.histogram(image, bins=256, range=(0, 256))[0] / image.size
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))

        return tuple[black_ratio, std, entropy]

    def is_blank(self, image: np.ndarray) -> Tuple[bool, float, str]:

        features = self._extract_features(image)

        probabilities = self.model.predict_proba(features)[0]

        blank_confidence = probabilities[1]

        is_blank = bool(blank_confidence >= self.confidence_threshold)

        reason = f"random_forest (confidence: {blank_confidence:.4f})"

        return is_blank, float(blank_confidence), reason
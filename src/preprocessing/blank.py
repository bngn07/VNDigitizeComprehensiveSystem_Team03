# ====================================================================
#       blank.py
# ====================================================================
# IMPORTS
# --------------------------------------------------------------------
from typing import Tuple
import numpy as np
import joblib
import cv2

# ====================================================================
# BLANK PAGE DETECTOR
# --------------------------------------------------------------------
#   - Detech when a page is blank or not.
#       + Do a binary threshold -> then count for black pixel
#       + Calculate for ratio for its fit on given upper and lower
#                           limit of black ratio
#       + If it's in the unsure area (+- given amount of ratio), then
#                           predict with Random Forest. 
#   
#   - Constructor:
#       + model_path: path to the random forest model (.joblib)
#       + threshold: lowest score to be considered blank
#       + lower: lower black ratio threshold
#       + upper: upper black ratio threshold
class BlankDetector():
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        lower: float = 0.005,
        upper: float = 0.95,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], received {threshold}")
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(f"0 <= lower < upper <= 1, received lower {lower}, upper={upper}")

        self.threshold = threshold
        self.lower = lower
        self.upper = upper
        self.model = joblib.load(model_path)

    @staticmethod
    def extract_blank_features(image: np.ndarray) -> np.ndarray:
        '''
            Manually extract an image's features by calculating
            its black_ratio, standard variation of all of the pixels,
            and its level of complex with Shannon Entropy.

            Args:
                - An image in np.ndarray (by cv2.imread)
            Return:
                - A list containing its feature, 
                    featuring black_ratio, std_val, entropy_val
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        black_ratio = float(np.sum(gray < 127) / gray.size)

        std_val = float(np.std(gray))

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten() / gray.size
        hist = hist[hist > 0]
        entropy_val = float(-np.sum(hist * np.log2(hist)))

        return np.array([black_ratio, std_val, entropy_val])

    def is_blank(self, image: np.ndarray) -> Tuple[bool, float, str]:
        '''
            Decide if the image is actually blank.
            
            First, find image's feature, then check for black ratio
                , if it's in the right condition, return and skip
                , if not, run RF model to predict.
            
            Args:
                - An image in np.ndarray (by cv2.imread)
            Return:
                - A list containing its feature, 
                    featuring is_blank, blank_score and detailed comment.
        '''
        features = self.extract_blank_features(image)
        black_ratio = float(features[0])

        # if black_ratio < self.lower:
        #     return True, 1.0, f"density_too_white (ratio={black_ratio:.4f})"
        # if black_ratio > self.upper:
        #     return True, 1.0, f"density_too_black (ratio={black_ratio:.4f})"

        probs = self.model.predict_proba(features.reshape(1, -1))[0]
        blank_score = float(probs[1])
        is_blank = blank_score >= self.threshold

        return is_blank, blank_score, f"rf_model (score={blank_score:.4f})"
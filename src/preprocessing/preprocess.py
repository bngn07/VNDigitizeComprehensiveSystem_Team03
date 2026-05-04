import numpy as np
import cv2

from .blank import BlankDetector
from .code import CodePreprocessor


# pipeline_config = {
#     "blank_detector": {
#         "model_path": "models/rf_blank_v1.joblib",
#         "threshold": 0.65,
#         "lower": 0.01,
#         "upper": 0.98
#     },
#     "code_detector": {
#         "types": ["QRCODE", "CODE128"]
#     },
#     "image_preprocessor": {
#         "gaussian_blur": {"kernel_size": (5, 5), "sigma_x": 0},
#         "enable_deskew": True,
#         "deskew": {"max_angle": 45},
#         "enable_autocrop": True,
#         "adaptive_threshold": {"max_value": 255, "block_size": 11, "c": 2},
#         "sharpening": {
#             "enable": True,
#             "kernel": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
#         }
#     }
# }


class Preprocessing:
    def __init__(self, config: dict = None):
        self.config = config if config is not None else {}

        blank_config = self.config.get("blank_detector", {})
        code_config = self.config.get("code_preprocessor", {})

        self.blank_detector = BlankDetector(**blank_config)
        self.code_preprocessor = CodePreprocessor(**code_config)

        self.cv_cfg = self.config.get("image_preprocessor", {})

    # ------------------------------------------------------------------
    # Helper steps
    # ------------------------------------------------------------------
    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale. If already 1-channel, returns a copy."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for denoising."""
        gb_cfg = self.cv_cfg.get("gaussian_blur", {})
        k_size = tuple(gb_cfg.get("kernel_size", (5, 5)))
        sigma = gb_cfg.get("sigma_x", 0)
        return cv2.GaussianBlur(gray, k_size, sigma)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect skew using the largest contour's minAreaRect and rotate the image.
        Returns the deskewed grayscale image.
        """
        if not self.cv_cfg.get("enable_deskew", True):
            return gray

        deskew_cfg = self.cv_cfg.get("deskew", {})
        max_angle = deskew_cfg.get("max_angle", 45)

        # Invert + Otsu binarization to highlight foreground
        inverted = cv2.bitwise_not(gray)
        _, thresh = cv2.threshold(
            inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Find contours and use the largest one
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return gray

        largest = max(contours, key=cv2.contourArea)
        angle = cv2.minAreaRect(largest)[-1]

        # Normalize angle to [-45, 45]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        # Clamp to configured maximum
        angle = max(min(angle, max_angle), -max_angle)

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _autocrop(self, image: np.ndarray) -> np.ndarray:
        """Crop empty borders using Otsu threshold + bounding box of non-zero pixels."""
        if not self.cv_cfg.get("enable_autocrop", True):
            return image

        _, thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        coords = cv2.findNonZero(thresh)
        if coords is None:
            cropped = image
        else:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = image[y:y + h, x:x + w]

        # Safety: ensure single-channel output
        if len(cropped.shape) == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        return cropped

    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Normalize background (yellowish paper, uneven lighting) via adaptive thresholding."""
        thresh_cfg = self.cv_cfg.get(
            "adaptive_threshold",
            {"max_value": 255, "block_size": 11, "c": 2},
        )
        return cv2.adaptiveThreshold(
            image,
            thresh_cfg["max_value"],
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            thresh_cfg["block_size"],
            thresh_cfg["c"],
        )

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen text using a 3x3 kernel."""
        sharp_cfg = self.cv_cfg.get("sharpening", {})
        if not sharp_cfg.get("enable", True):
            return image

        kernel_val = sharp_cfg.get(
            "kernel", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        )
        kernel = np.array(kernel_val)
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def _build_metadata(is_blank: bool, blank_score, comment, qr_list) -> dict:
        return {
            "status": "success",
            "is_blank": is_blank,
            "blank_score": blank_score,
            "comment": comment,
            "qr_codes": qr_list,
        }

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def _process(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        # 1. Grayscale
        gray = self._to_grayscale(image)

        # 2. Blank check (early exit)
        is_blank, blank_score, comment = self.blank_detector.is_blank(gray)
        if is_blank:
            return image, self._build_metadata(True, blank_score, comment, [])

        # 3. Denoise
        blurred = self._denoise(gray)

        # 4. Deskew (operates on denoised grayscale)
        deskewed = self._deskew(blurred)

        # 5. Autocrop
        cropped = self._autocrop(deskewed)

        # 6. QR / Barcode detection on the cleanest grayscale version
        qr_list = self.code_preprocessor.detect(cropped)

        # 7. Adaptive threshold (background normalization)
        normalized = self._adaptive_threshold(cropped)

        # 8. Sharpening
        sharpened = self._sharpen(normalized)

        metadata = self._build_metadata(False, blank_score, comment, qr_list)
        return sharpened, metadata

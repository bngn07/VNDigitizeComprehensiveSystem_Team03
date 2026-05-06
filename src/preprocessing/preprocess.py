from dataclasses import dataclass

import numpy as np
import cv2

from .blank import BlankDetector
from .code import CodeDetector
from .rotate import RotationDetector

'''
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
'''

@dataclass
class PreprocessResult:
    image: np.ndarray
    metadata: dict

    def __str__(self):
        parts = []

        # Image info
        parts.append("IMAGE")
        if self.image is not None:
            h, w = self.image.shape[:2]
            c = self.image.shape[2] if len(self.image.shape) > 2 else 1
            parts.append(f"  size     : {w} x {h}")
            parts.append(f"  channels : {c}")
            parts.append(f"  dtype    : {self.image.dtype}")
        else:
            parts.append("  None")

        # Metadata
        status = self.metadata.get("status")
        is_blank = self.metadata.get("is_blank", False)
        confidence = self.metadata.get("confidence", 0.0)
        qr_codes = self.metadata.get("qr_codes", [])

        parts.append("\nMETADATA")
        parts.append(f"  status     : {status}")
        parts.append(f"  is_blank   : {is_blank}")
        parts.append(f"  confidence : {confidence:.4f}")
        parts.append(f"  qr_count   : {len(qr_codes)}")

        # QR details
        if qr_codes:
            parts.append("\nQR OBJECTS")
            for i, qr in enumerate(qr_codes, 1):
                parts.append(f"  [{i}]")
                
                if hasattr(qr, '__dict__'):
                    data = qr.__dict__
                elif hasattr(qr, '__slots__'):
                    data = {s: getattr(qr, s) for s in qr.__slots__}
                elif isinstance(qr, dict):
                    data = qr
                else:
                    data = {"value": str(qr)}

                for k, v in data.items():
                    parts.append(f"    {k}: {v}")

        return "\n".join(parts)

class Preprocessing:
    def __init__(self, config: dict = None):
        self.config = config if config is not None else {}

        blank_config = self.config.get("blank_detector", {})
        code_config = self.config.get("code_preprocessor", {})
        osd_config = self.config.get("tesseract", {})

        self.blank_detector = BlankDetector(**blank_config)
        self.code_preprocessor = CodeDetector(**code_config)
        self.rotation_detector = RotationDetector(**osd_config)

        self.cv_cfg = self.config.get("image_preprocessor", {})

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
        Detect skew using the minAreaRect of ALL foreground pixels.
        Returns the deskewed grayscale image.
        """
        if not self.cv_cfg.get("enable_deskew", True):
            return gray

        deskew_cfg = self.cv_cfg.get("deskew", {})
        max_angle = deskew_cfg.get("max_angle", 45)

        gray_inv = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Tìm tất cả các tọa độ pixel có giá trị > 0
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # Điều chỉnh góc xoay
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Xoay ảnh
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

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

    # def _transparent_to_white(self, image: np.ndarray) -> np.ndarray:
    #     if image.ndim != 3 or image.shape[2] != 4:
    #         return image

    #     bgr = image[:, :, :3].astype(np.float32)
    #     alpha = image[:, :, 3].astype(np.float32) / 255.0
    #     alpha = alpha[:, :, np.newaxis]

    #     white = np.full_like(bgr, 255.0)

    #     composited = bgr * alpha + white * (1.0 - alpha)
    #     return np.clip(composited, 0, 255).astype(np.uint8)

    @staticmethod
    def _build_metadata(is_blank: bool, confidence, comment, qr_list) -> dict:
        return {
            "status": "success",
            "is_blank": is_blank,
            "confidence": confidence,
            "comment": comment,
            "qr_codes": qr_list,
        }

    def _process(self, image: np.ndarray) -> PreprocessResult:
        # # Make transparent pixels white.       
        # image = self._transparent_to_white(image)

        # 1. Grayscale
        gray = self._to_grayscale(image)

        # 2. Blank check
        blank_result = self.blank_detector.is_blank(gray)
        if blank_result.is_blank:
            return PreprocessResult(
                image=image,
                metadata=self._build_metadata(
                    True, blank_result.confidence, blank_result.comment, []
                ),
            )

        # 3. Orientation correction
        oriented = self.rotation_detector._orient(gray)

        # 4. Light denoise
        blurred = self._denoise(oriented)

        # 5. Deskew
        deskewed = self._deskew(blurred)

        # 6. Autocrop
        cropped = self._autocrop(deskewed)

        # 7. QR / Barcode detection
        qr_results = self.code_preprocessor.detect(cropped)

        # 8. Adaptive threshold
        normalized = self._adaptive_threshold(cropped)

        # 9. Sharpening
        sharpened = self._sharpen(normalized)

        metadata = self._build_metadata(
            False, blank_result.confidence, blank_result.comment, qr_results
        )
        return PreprocessResult(image=sharpened, metadata=metadata)
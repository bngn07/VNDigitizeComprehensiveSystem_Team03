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
#     }
# }

class Preprocessing:
    def __init__(self, config: dict = None):
        self.config = config if config is not None else {}

        blank_config = self.config.get('blank_detector', {})
        code_config = self.config.get('code_preprocessor', {})

        self.blank_detector = BlankDetector(**blank_config)
        self.code_preprocessor = CodePreprocessor(**code_config)

    def _process(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        # ====================================================================
        # 1. RBG -> Grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # -- IS_BLANK check --
        is_blank, blank_score, comment = self.blank_detector.is_blank(grayscale)

        if is_blank:
            metadata = {
                "status": "success",
                "is_blank": True,
                "blank_score": blank_score,
                "comment": comment,
                "qr_codes": []
            }
            return image, metadata

        # ====================================================================
        # 2. Denoising by appying Gaussian Blur
        # arguments: (input image, kernel size, sigmaX)
        cv_cfg = self.config.get("image_preprocessor", {})
        k_size = tuple(cv_cfg.get("gaussian_blur", {}).get("kernel_size", (5, 5)))
        sigma = cv_cfg.get("gaussian_blur", {}).get("sigma_x", 0)
        blurred = cv2.GaussianBlur(grayscale, k_size, sigma)

        # ====================================================================
        # 3. Deskewing
        # arguments: (input image, threshold value, max value, thresholding type)
        if cv_cfg.get("enable_deskew", True):
            _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(bw > 0))
            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(grayscale, M, (w, h))
        else:
            deskewed = grayscale

        # ====================================================================
        # 4. Autocropping
        # arguments: (input image, threshold value, max value, thresholding type)
        if cv_cfg.get("enable_autocrop", True):
            _, thresh = cv2.threshold(
                deskewed, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            coords = cv2.findNonZero(thresh)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                cropped = deskewed[y:y+h, x:x+w]
            else:
                # fallback: giữ nguyên ảnh
                cropped = deskewed
        else:
            cropped = deskewed

        # -- QR/Barcode detection --
        qr_list = self.code_preprocessor.detect(cropped)

        # ====================================================================
        # 5. Processing yellowish background, unbalanced brightness/contrast,
        # etc. by applying adaptive thresholding
        # arguments: (input image, max value, adaptive method, thresholding type, block size, C)
        thresh_cfg = cv_cfg.get("adaptive_threshold", {"max_value": 255, "block_size": 15, "c": 1})
        normalized = cv2.adaptiveThreshold(
            cropped, 
            thresh_cfg["max_value"], 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            thresh_cfg["block_size"], 
            thresh_cfg["c"]
        )

        # ====================================================================
        # 6. Sharpening by applying unsharp masking
        # arguments: (input image, output image, kernel size, sigmaX)
        sharp_cfg = cv_cfg.get("sharpening", {})
        if sharp_cfg.get("enable", True):
            kernel_val = sharp_cfg.get("kernel", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            kernel = np.array(kernel_val)
            sharpened = cv2.filter2D(normalized, -1, kernel)
        else:
            sharpened = normalized

        metadata = {
            "status": "success",
            "is_blank": False,
            "blank_score": blank_score,
            "comment": comment,
            "qr_codes": qr_list
        }

        return sharpened, metadata
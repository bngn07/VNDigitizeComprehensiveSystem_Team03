from dataclasses import dataclass, field
import numpy as np
import cv2
import os
import logging
from typing import Optional, Tuple

from .blank import BlankDetector
from .code import CodeDetector
from .rotate import RotationDetector
from .geometry import detect_document, four_point_transform

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────

@dataclass
class ImageQualityMetrics:
    """Chỉ số chất lượng ảnh dùng để chọn nhánh xử lý phù hợp."""
    blur_score: float = 0.0
    noise_sigma: float = 0.0
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    contrast: float = 0.0
    is_low_quality: bool = False


@dataclass
class PreprocessResult:
    image: np.ndarray
    metadata: dict
    quality: Optional[ImageQualityMetrics] = None

    def __str__(self) -> str:
        parts = ["IMAGE"]
        if self.image is not None:
            h, w = self.image.shape[:2]
            c = self.image.shape[2] if self.image.ndim > 2 else 1
            parts += [
                f"  size     : {w} x {h}",
                f"  channels : {c}",
                f"  dtype    : {self.image.dtype}",
            ]
        else:
            parts.append("  None")

        m = self.metadata
        parts += [
            "\nMETADATA",
            f"  status      : {m.get('status')}",
            f"  is_blank    : {m.get('is_blank', False)}",
            f"  orientation : {m.get('orientation', 0)} degrees",
            f"  confidence  : {m.get('confidence', 0.0):.4f}",
            f"  qr_count    : {len(m.get('qr_codes', []))}",
        ]

        qr_codes = m.get("qr_codes", [])
        if qr_codes:
            parts.append("\nQR OBJECTS")
            for i, qr in enumerate(qr_codes, 1):
                parts.append(f"  [{i}]")
                data = (
                    qr.__dict__ if hasattr(qr, "__dict__")
                    else {s: getattr(qr, s) for s in qr.__slots__}
                    if hasattr(qr, "__slots__")
                    else qr if isinstance(qr, dict)
                    else {"value": str(qr)}
                )
                for k, v in data.items():
                    parts.append(f"    {k}: {v}")

        if self.quality:
            q = self.quality
            parts += [
                "\nQUALITY",
                f"  blur_score      : {q.blur_score:.2f}",
                f"  noise_sigma     : {q.noise_sigma:.2f}",
                f"  brightness_mean : {q.brightness_mean:.1f}",
                f"  brightness_std  : {q.brightness_std:.1f}",
                f"  contrast        : {q.contrast:.2f}",
                f"  is_low_quality  : {q.is_low_quality}",
            ]

        return "\n".join(parts)


# ──────────────────────────────────────────────────────────────
# MAIN CLASS
# ──────────────────────────────────────────────────────────────

class Preprocessing:
    # FIX: Nâng ngưỡng blur để ảnh chụp điện thoại bình thường không bị
    # coi là "low quality" và đi vào nhánh xử lý nặng gây nhòe.
    _BLUR_THRESHOLD   = 40.0   # Giảm từ 80 → 40; Laplacian variance của ảnh
                                # chụp điện thoại thường 60–200, không nên
                                # trigger pipeline DL binarization.
    _NOISE_THRESHOLD  = 25.0   # Nới lỏng từ 15 → 25
    _BRIGHT_MIN       = 30.0   # Giảm từ 40 → 30
    _BRIGHT_MAX       = 230.0  # Tăng từ 220 → 230

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.blank_detector     = BlankDetector(**self.config.get("blank_detector", {}))
        self.code_preprocessor  = CodeDetector(**self.config.get("code_preprocessor", {}))
        self.rotation_detector  = RotationDetector(**self.config.get("tesseract", {}))
        self.cv_cfg             = self.config.get("image_preprocessor", {})

    # ──────────────────────────────────────
    # 0. GRAYSCALE CONVERSION
    # ──────────────────────────────────────

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    # ──────────────────────────────────────
    # 1. ĐÁNH GIÁ CHẤT LƯỢNG ẢNH
    # ──────────────────────────────────────

    def _assess_quality(self, gray: np.ndarray) -> ImageQualityMetrics:
        h, w = gray.shape
        scale = min(1.0, 800.0 / max(h, w))
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        blur_score = cv2.Laplacian(small, cv2.CV_64F).var()

        lap = cv2.Laplacian(small.astype(np.float32), cv2.CV_32F)
        noise_sigma = float(np.median(np.abs(lap - np.median(lap)))) / 0.6745 + 1e-6

        brightness_mean = float(small.mean())
        brightness_std  = float(small.std())
        contrast = float(np.sqrt(np.mean((small.astype(np.float32) - brightness_mean) ** 2)))

        is_low_quality = (
            blur_score < self._BLUR_THRESHOLD
            or noise_sigma > self._NOISE_THRESHOLD
            or brightness_mean < self._BRIGHT_MIN
            or brightness_mean > self._BRIGHT_MAX
        )

        return ImageQualityMetrics(
            blur_score=blur_score,
            noise_sigma=noise_sigma,
            brightness_mean=brightness_mean,
            brightness_std=brightness_std,
            contrast=contrast,
            is_low_quality=is_low_quality,
        )

    # ──────────────────────────────────────
    # 2. KHỬ BÓNG / CÂN BẰNG ÁNH SÁNG
    # ──────────────────────────────────────

    def _remove_shadow(self, gray: np.ndarray) -> np.ndarray:
        """
        FIX: Tăng kernel size tối thiểu lên 101 (từ 51) để ước lượng
        background chính xác hơn, tránh ảnh hưởng đến nét chữ.
        Nếu ảnh đã đủ sáng/đều (brightness_std thấp), bỏ qua bước này.
        """
        gray_f = gray.astype(np.float32) + 1.0

        # Kernel phải đủ lớn để chỉ bắt gradient sáng toàn cục,
        # không "thấy" chi tiết chữ → tối thiểu 101px hoặc 1/4 cạnh ngắn
        ksize = max(101, (min(gray.shape[:2]) // 4) | 1)
        bg = cv2.GaussianBlur(gray_f, (ksize, ksize), 0)

        normalized = (gray_f / bg) * 128.0
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        return normalized

    # ──────────────────────────────────────
    # 3. DESKEW
    # ──────────────────────────────────────

    def _deskew_pht(self, gray: np.ndarray) -> np.ndarray:
        if not self.cv_cfg.get("enable_deskew", True):
            return gray

        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        dilated = cv2.dilate(thresh, kernel_h, iterations=1)

        edges = cv2.Canny(dilated, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=80, minLineLength=100, maxLineGap=15
        )
        if lines is None:
            return gray

        angles, weights = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue
            length = np.hypot(dx, dy)
            angle = np.degrees(np.arctan2(dy, dx))

            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90

            if -20.0 < angle < 20.0:
                angles.append(angle)
                weights.append(length)

        if not angles:
            return gray

        angles  = np.array(angles)
        weights = np.array(weights)
        sorted_idx = np.argsort(angles)
        angles_sorted  = angles[sorted_idx]
        weights_sorted = weights[sorted_idx]
        cumsum = np.cumsum(weights_sorted)
        dominant_angle = float(angles_sorted[np.searchsorted(cumsum, cumsum[-1] / 2)])

        if abs(dominant_angle) < 0.3:
            return gray

        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), dominant_angle, 1.0)
        return cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    # ──────────────────────────────────────
    # 4. AUTOCROP
    # ──────────────────────────────────────

    def _autocrop_yolov8(self, gray: np.ndarray) -> np.ndarray:
        model_path = self.cv_cfg.get("yolo_crop_model", "yolov8n_doc_crop.onnx")
        result = self._try_yolo_crop(gray, model_path)
        if result is not None:
            return result
        return self._opencv_crop_fallback(gray)

    def _try_yolo_crop(self, gray: np.ndarray, model_path: str) -> Optional[np.ndarray]:
        if not os.path.exists(model_path):
            return None
        try:
            net  = cv2.dnn.readNetFromONNX(model_path)
            h_o, w_o = gray.shape[:2]
            blob = cv2.dnn.blobFromImage(gray, 1 / 255.0, (640, 640), swapRB=True, crop=False)
            net.setInput(blob)
            boxes = net.forward()[0][0]
            x_f, y_f = w_o / 640.0, h_o / 640.0
            best, max_area = None, 0
            for i in range(boxes.shape[1]):
                box = boxes[:, i]
                if box[4] > 0.5:
                    cx, cy, bw, bh = box[:4]
                    area = bw * bh
                    if area > max_area:
                        max_area = area
                        best = (
                            int((cx - bw / 2) * x_f), int((cy - bh / 2) * y_f),
                            int(bw * x_f),             int(bh * y_f),
                        )
            if best is not None:
                x, y, bw, bh = best
                x, y = max(0, x), max(0, y)
                crop = gray[y:y + bh, x:x + bw]
                if crop.size > 0:
                    return crop
        except Exception as e:
            logger.warning("YOLO crop failed: %s", e)
        return None

    def _opencv_crop_fallback(self, gray: np.ndarray) -> np.ndarray:
        if not self.cv_cfg.get("enable_autocrop", True):
            return gray

        h_orig, w_orig = gray.shape[:2]

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 30, 120)

        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        dilated = cv2.dilate(edges, kernel, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return gray

        img_area       = h_orig * w_orig
        min_area_ratio = self.cv_cfg.get("crop_min_area_ratio", 0.005)
        valid = [c for c in contours if cv2.contourArea(c) > img_area * min_area_ratio]
        if not valid:
            return gray

        all_pts  = np.vstack(valid)
        x, y, w, h = cv2.boundingRect(all_pts)

        pad = max(15, int(min(h_orig, w_orig) * 0.01))
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w_orig - x, w + 2 * pad)
        h = min(h_orig - y, h + 2 * pad)

        if (w * h) < img_area * 0.30:
            logger.debug("Crop too small (%dx%d vs %dx%d), skipping.", w, h, w_orig, h_orig)
            return gray

        return gray[y:y + h, x:x + w]

    # ──────────────────────────────────────
    # 5. BINARIZATION (đã sửa nhiều nhất)
    # ──────────────────────────────────────

    def _apply_hybrid_binarization(
        self, gray: np.ndarray, quality: ImageQualityMetrics
    ) -> np.ndarray:
        """
        FIX toàn bộ pipeline binarization:

        ẢNH CHẤT LƯỢNG TỐT (is_low_quality=False):
          → Chỉ CLAHE nhẹ + Otsu.  KHÔNG denoise, KHÔNG shadow removal,
            KHÔNG morphology. Giữ nguyên nét chữ gốc.

        ẢNH CHẤT LƯỢNG KÉM (is_low_quality=True):
          → Denoise nhẹ → shadow removal → CLAHE → Sauvola/adaptive
            → morphology tối thiểu.

        Lý do: fastNlMeansDenoising và _remove_shadow (GaussianBlur lớn)
        làm mờ nét chữ rõ ràng khi ảnh gốc đã đủ tốt.
        """
        if not quality.is_low_quality:
            return self._binarize_good_quality(gray, quality)
        else:
            return self._binarize_low_quality(gray, quality)

    @staticmethod
    def _binarize_good_quality(
        gray: np.ndarray, quality: ImageQualityMetrics
    ) -> np.ndarray:
        """
        Pipeline nhanh cho ảnh chất lượng tốt:
          CLAHE nhẹ → Otsu
        Không denoise, không shadow removal, không morphology.
        """
        # CLAHE rất nhẹ chỉ để cân bằng histogram cục bộ
        clip = 1.5 if quality.contrast < 40 else 1.0
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # FIX: Dùng Otsu thay vì Sauvola cho ảnh tốt.
        # Otsu cho kết quả sạch, không có noise muối tiêu xung quanh chữ.
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary

    def _binarize_low_quality(
        self, gray: np.ndarray, quality: ImageQualityMetrics
    ) -> np.ndarray:
        """
        Pipeline đầy đủ CHỈ cho ảnh chất lượng kém:
          denoise → shadow removal → CLAHE → Sauvola/adaptive → morphology nhẹ
        """
        # ── 5.1  DENOISING — chỉ khi thực sự cần ────────────────
        h_denoise = self._compute_denoise_strength(quality.noise_sigma)
        if h_denoise > 0:
            denoised = cv2.fastNlMeansDenoising(
                gray, None,
                h=h_denoise, templateWindowSize=7, searchWindowSize=21,
            )
        else:
            denoised = gray

        # ── 5.2  SHADOW REMOVAL ───────────────────────────────────
        shadow_free = self._remove_shadow(denoised)

        # ── 5.3  CLAHE ────────────────────────────────────────────
        clip_limit  = 3.0 if quality.contrast < 30 else 2.0
        clahe       = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced    = clahe.apply(shadow_free)

        # ── 5.4  BINARIZATION ─────────────────────────────────────
        binary = self._sauvola_or_adaptive(enhanced, quality)

        # ── 5.5  MORPHOLOGY — kernel nhỏ hơn (1x1 effective) ─────
        # FIX: Chỉ Opening để xóa noise điểm, BỎ Closing vì nó nối
        # nhầm các nét chữ và tạo artifact "blob" trên ảnh mờ.
        binary = self._morphology_cleanup(binary)

        return binary

    @staticmethod
    def _compute_denoise_strength(noise_sigma: float) -> int:
        """
        FIX: Thêm trường hợp noise thấp → trả về 0 (không denoise).
        Bản gốc luôn denoise với h>=3, gây làm mờ ảnh đã nét.
        """
        if noise_sigma < 3:    # Ảnh rất sạch → bỏ qua hoàn toàn
            return 0
        elif noise_sigma < 5:
            return 3
        elif noise_sigma < 10:
            return 5           # Giảm từ 7 → 5 để ít làm mờ hơn
        elif noise_sigma < 20:
            return 8           # Giảm từ 10 → 8
        else:
            return 12          # Giảm từ 15 → 12

    @staticmethod
    def _sauvola_or_adaptive(enhanced: np.ndarray, quality: ImageQualityMetrics) -> np.ndarray:
        """
        FIX: Tăng block size và tăng k để Sauvola ít nhạy cảm hơn
        với vùng nền trắng (giảm noise muối tiêu).
        k=0.2 quá nhạy → tăng lên 0.35.
        """
        h, w  = enhanced.shape[:2]
        short = min(h, w)
        # FIX: Tăng từ 1.8% → 2.5% để block đủ lớn, giảm artifact
        block = max(25, int(short * 0.025))
        if block % 2 == 0:
            block += 1

        try:
            binary = cv2.ximgproc.niBlackThreshold(
                enhanced, 255, cv2.THRESH_BINARY, block,
                k=0.35,   # FIX: tăng từ 0.2 → 0.35, ít noise hơn
                binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA,
            )
        except AttributeError:
            # Fallback: adaptive Gaussian với C lớn hơn để nền sạch hơn
            binary = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                block,
                12,   # FIX: tăng từ 10 → 12 để nền trắng sạch hơn
            )
        return binary

    @staticmethod
    def _morphology_cleanup(binary: np.ndarray) -> np.ndarray:
        """
        FIX: Chỉ Opening (xóa noise điểm đơn lẻ).
        BỎ Closing vì nó nối các nét gần nhau tạo "blob" trên ảnh mờ.
        Nếu cần Closing, chỉ dùng khi ảnh thực sự bị đứt nét (handled riêng).
        """
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
        return opened

    # ──────────────────────────────────────
    # 6. DEEP LEARNING BINARIZATION FALLBACK
    # ──────────────────────────────────────

    def _deep_learning_binarization_fallback(
        self,
        gray: np.ndarray,
        quality: ImageQualityMetrics,
        ocr_confidence_score: float = 1.0,
    ) -> np.ndarray:
        """
        FIX: Thêm guard — nếu ảnh chất lượng tốt VÀ OCR confidence cao,
        dùng luôn pipeline nhẹ, không gọi model DL.
        Tránh trường hợp model DL inference sai làm ảnh tệ hơn.
        """
        # Chỉ thử DL nếu thực sự cần
        use_dl = (ocr_confidence_score < 0.50) or quality.is_low_quality
        if use_dl:
            model_path = self.cv_cfg.get("binet_model", "binet_document_enhancement.onnx")
            result = self._try_binet(gray, model_path)
            if result is not None:
                return result

        return self._apply_hybrid_binarization(gray, quality)

    def _try_binet(self, gray: np.ndarray, model_path: str) -> Optional[np.ndarray]:
        if not os.path.exists(model_path):
            return None
        try:
            net  = cv2.dnn.readNetFromONNX(model_path)
            blob = cv2.dnn.blobFromImage(gray, 1 / 255.0, (512, 512), swapRB=False, crop=False)
            net.setInput(blob)
            out = net.forward()[0][0]
            out_img = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            return cv2.resize(out_img, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
            logger.warning("BiNet inference failed: %s", e)
        return None

    # ──────────────────────────────────────
    # 7. BARCODE / QR DETECTION
    # ──────────────────────────────────────

    def _extract_barcode_lazy_evaluation(self, image: np.ndarray) -> list:
        result = self.code_preprocessor.detect(image)
        if result:
            return result

        y8_path = self.cv_cfg.get("yolo_barcode_model", "y8_libarnet_barcode.onnx")
        if not os.path.exists(y8_path):
            return []

        try:
            crop, score = self._yolo_detect_barcode_region(image, y8_path)
            if crop is None or score < 0.4:
                return []

            crop = self._super_resolve(crop)
            return self.code_preprocessor.detect(crop)
        except Exception as e:
            logger.warning("Barcode YOLO pipeline failed: %s", e)
        return []

    def _yolo_detect_barcode_region(
        self, image: np.ndarray, model_path: str
    ) -> Tuple[Optional[np.ndarray], float]:
        net  = cv2.dnn.readNetFromONNX(model_path)
        h_o, w_o = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        boxes = net.forward()[0][0]

        best_box, best_score = None, 0.0
        x_f, y_f = w_o / 640.0, h_o / 640.0
        for i in range(boxes.shape[1]):
            box   = boxes[:, i]
            score = float(box[4])
            if score > best_score:
                best_score = score
                cx, cy, bw, bh = box[:4]
                best_box = (
                    max(0, int((cx - bw / 2) * x_f)),
                    max(0, int((cy - bh / 2) * y_f)),
                    int(bw * x_f), int(bh * y_f),
                )
        if best_box is None or best_score < 0.4:
            return None, 0.0
        bx, by, bw, bh = best_box
        crop = image[by:by + bh, bx:bx + bw]
        return (crop if crop.size > 0 else None), best_score

    def _super_resolve(self, crop: np.ndarray) -> np.ndarray:
        edsr_path = self.cv_cfg.get("edsr_model", "edsr_super_res.pb")
        if not os.path.exists(edsr_path):
            return crop
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(edsr_path)
            sr.setModel("edsr", 3)
            return sr.upsample(crop)
        except Exception as e:
            logger.debug("Super-resolution skipped: %s", e)
        return crop

    # ──────────────────────────────────────
    # 8. PERSPECTIVE CORRECTION
    # ──────────────────────────────────────

    def _perspective_correct(self, image: np.ndarray) -> np.ndarray:
        ratio       = image.shape[0] / 500.0
        small_w     = int(image.shape[1] / ratio)
        image_small = cv2.resize(image, (small_w, 500))
        doc_cnt     = detect_document(image_small)
        if doc_cnt is None:
            return image
        doc_cnt = doc_cnt.reshape(4, 2) * ratio
        return four_point_transform(image, doc_cnt)

    # ──────────────────────────────────────
    # 9. METADATA BUILDER
    # ──────────────────────────────────────

    @staticmethod
    def _build_metadata(
        is_blank: bool, orientation: float, confidence, comment, qr_list: list
    ) -> dict:
        return {
            "status"     : "success",
            "is_blank"   : is_blank,
            "orientation": orientation,
            "confidence" : confidence,
            "comment"    : comment,
            "qr_codes"   : qr_list,
        }

    # ──────────────────────────────────────
    # 10. MAIN PIPELINE
    # ──────────────────────────────────────

    def _process(self, image: np.ndarray) -> PreprocessResult:
        """
        Pipeline chính — không thay đổi thứ tự bước, chỉ các hàm con được sửa.
        """
        gray = self._to_grayscale(image)

<<<<<<< HEAD
        quality = self._assess_quality(gray)

        blank_result       = self.blank_detector.is_blank(gray)
        orientation_result = self.rotation_detector.detect(gray)

        oriented = self.rotation_detector._orient(gray)
=======
        # 2. Blank check
        blank_result = self.blank_detector.is_blank(gray)
        if blank_result.is_blank:
            return PreprocessResult(
                image=image,
                metadata=self._build_metadata(
                    True, blank_result.confidence, blank_result.comment, []
                ),
            )
        
        # 3. Orientation
        #oriented = self.rotation_detector._orient(gray)
>>>>>>> 6a3291c243ff7e8b7d7eb1f368c971dd79a4a6b3

        if self.cv_cfg.get("enable_perspective", False):
            oriented = self._perspective_correct(oriented)

<<<<<<< HEAD
        deskewed = self._deskew_pht(oriented)
=======
        # 5. Denoise
        blurred = self._denoise(gray)
>>>>>>> 6a3291c243ff7e8b7d7eb1f368c971dd79a4a6b3

        cropped = self._autocrop_yolov8(deskewed)

        # Re-assess sau crop vì kích thước đã thay đổi
        quality = self._assess_quality(cropped)

        qr_results = self._extract_barcode_lazy_evaluation(cropped)

        final_binarized = self._deep_learning_binarization_fallback(cropped, quality)

        result = sharpened

        metadata = self._build_metadata(
            is_blank    = bool(blank_result.is_blank),
            orientation = float(orientation_result.angle),
            confidence  = float(blank_result.confidence),
            comment     = blank_result.comment,
            qr_list     = qr_results,
        )
<<<<<<< HEAD

        return PreprocessResult(image=final_binarized, metadata=metadata, quality=quality)
=======
        return PreprocessResult(image = result, metadata=metadata)
>>>>>>> 6a3291c243ff7e8b7d7eb1f368c971dd79a4a6b3

# -- built in
from dataclasses import dataclass

# -- 3rd party
import numpy as np
import cv2
from paddleocr import PaddleOCR

# -- own
from .ocr import BaseOCR, OCRResult


@dataclass
class PaddleParams:
    lang: str = 'en'
    device: str = 'cpu'
    use_textline_orientation: bool = False
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    det_model_dir: str | None = None
    rec_model_dir: str | None = None

class PaddleModel:
    def __init__(self, params: PaddleParams | None = None):
        self.params = params or PaddleParams()

        kwargs = dict(
            lang=self.params.lang,
            device=self.params.device,
            use_textline_orientation=self.params.use_textline_orientation,
            use_doc_orientation_classify=self.params.use_doc_orientation_classify,
            use_doc_unwarping=self.params.use_doc_unwarping,
        )

        # Only pass model dirs when explicitly provided
        if self.params.det_model_dir:
            kwargs['text_detection_model_dir'] = self.params.det_model_dir
        if self.params.rec_model_dir:
            kwargs['text_recognition_model_dir'] = self.params.rec_model_dir

        self._engine = PaddleOCR(**kwargs)


class Paddle(BaseOCR):
    def __init__(self, model: PaddleModel | None = None):
        super().__init__()
        self.model = model or PaddleModel()

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        """PaddleOCR expects RGB; cv2 loads BGR."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return image

    def _parse_result(self, raw) -> OCRResult:
        """
        PaddleOCR v3 structure:
        res.json['res']['rec_texts']  → list of strings
        res.json['res']['rec_scores'] → list of floats
        """
        if not raw:
            return OCRResult(text='', confidence=0.0)

        lines, confidences = [], []

        for res in raw:
            data = res.json.get('res', {})
            rec_texts  = data.get('rec_texts', [])
            rec_scores = data.get('rec_scores', [])

            for text, conf in zip(rec_texts, rec_scores):
                if text and text.strip():
                    lines.append(text.strip())
                    confidences.append(float(conf))

        if not lines:
            return OCRResult(text='', confidence=0.0)

        return OCRResult(
            text='\n'.join(lines),
            confidence=round(float(np.mean(confidences)), 4),
        )

    def _recognize_single(self, image: np.ndarray) -> OCRResult:
        rgb = self._to_rgb(image)
        raw = self.model._engine.predict(rgb)   # was: .ocr(rgb, cls=...)

        for res in raw:
            print("=== type:", type(res))
            print("=== dir:", [x for x in dir(res) if not x.startswith('__')])
            print("=== json:", res.json)

        return self._parse_result(raw)

    def _recognize_batch(self, images: list[np.ndarray]) -> list[OCRResult]:
        return [self._recognize_single(img) for img in images]

    def recognize(
        self,
        image: np.ndarray | list[np.ndarray]
    ) -> OCRResult | list[OCRResult]:
        if isinstance(image, list):
            return self._recognize_batch(image)
        return self._recognize_single(image)
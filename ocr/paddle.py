# -- built in
from dataclasses import (dataclass, asdict)

# -- 3rd party
import numpy as np
import cv2
from paddleocr import PaddleOCR

# -- own
from .ocr import BaseOCR, OCRResult, TextBlock


@dataclass
class PaddleParams:
    lang: str = 'vie'
    device: str = 'cpu'
    ocr_version: str = 'PP-OCRv5'

    use_textline_orientation: bool = False
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False

    det_model_dir: str | None = None
    rec_model_dir: str | None = None

class PaddleModel:
    def __init__(self, params: PaddleParams | None = None):
        self.params = params or PaddleParams()

        kwargs = {k: v for k, v in asdict(self.params).items() if v is not None}

        self._engine = PaddleOCR(**kwargs)

class Paddle(BaseOCR):
    def __init__(self, model: PaddleModel | None = None):
        super().__init__()
        self.model = model or PaddleModel()

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return image

    def _parse_result(self, raw) -> OCRResult:
        if not raw:
            return OCRResult(texts=[])

        blocks = []

        for res in raw:
            data = res.json.get('res', {})
            rec_texts  = data.get('rec_texts', [])
            rec_scores = data.get('rec_scores', [])
            
            dt_polys = data.get('dt_polys', [])

            if not dt_polys and rec_texts:
                dt_polys = [[] for _ in rec_texts]

            for text, conf, poly in zip(rec_texts, rec_scores, dt_polys):
                if text and text.strip():
                    polygon = [(int(p[0]), int(p[1])) for p in poly] if poly else []
                    
                    block = TextBlock(
                        text=text.strip(),
                        bounding_polygon=polygon,
                        confidence=float(conf)
                    )
                    blocks.append(block)

        return OCRResult(texts=blocks)

    def _recognize_single(self, image: np.ndarray) -> OCRResult:
        rgb = self._to_rgb(image)
        raw = self.model._engine.predict(rgb)

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
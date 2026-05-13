from dataclasses import dataclass

import numpy as np

from src.preprocessing.preprocess   import Preprocessing, PreprocessResult
from src.postprocessing.postprocess import Postprocessing
from ocr.ocr                        import BaseOCR, OCRResult


@dataclass
class DigitizeResult:
    image:      np.ndarray
    preprocess: PreprocessResult
    ocr:        OCRResult


class Digitize:
    def __init__(self, ocr: BaseOCR, config: dict = None) -> None:
        config = config or {}
        self.preprocess  = Preprocessing(config.get("preprocessing", {}))
        self.postprocess = Postprocessing(config.get("postprocessing", {}))
        self.ocr         = ocr

    def _digitize_single(self, image: np.ndarray) -> DigitizeResult:
        pre    = self.preprocess.process(image)
        raw    = self.ocr.recognize(pre.image)
        result = self.postprocess.process(raw)
        return DigitizeResult(
            image = pre.image,
            preprocess = pre, 
            ocr = result
            )

    def _digitize_batch(self, images: list[np.ndarray]) -> list[DigitizeResult]:
        pres    = [self.preprocess.process(img) for img in images]
        raws    = self.ocr.recognize([p.image for p in pres])
        results = [self.postprocess.process(r) for r in raws]
        return [DigitizeResult(preprocess=p, ocr=r) for p, r in zip(pres, results)]

    def digitize(
        self,
        image: np.ndarray | list[np.ndarray],
    ) -> DigitizeResult | list[DigitizeResult]:
        if isinstance(image, list):
            return self._digitize_batch(image)
        return self._digitize_single(image)
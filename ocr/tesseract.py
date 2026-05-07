# -- built in
from dataclasses import dataclass

# -- 3rd party
import numpy as np
import pytesseract

# -- own
from .ocr import BaseOCR, OCRResult

@dataclass
class TesseractParams:
    lang: str = 'vie'
    psm: int = 3
    oem: int = 3

class TesseractModel:
    def __init__(self, params: TesseractParams | None = None):
        self.params = params or TesseractParams()

        self.config_string = f"--oem {self.params.oem} --psm {self.params.psm}"

class TesseractOCR(BaseOCR):
    def __init__(self, model: TesseractModel | None = None):
        super().__init__()
        self.model = model or TesseractModel()

    def _recognize_single(self, image: np.ndarray) -> OCRResult:
        text = pytesseract.image_to_string(
            image, 
            lang=self.model.params.lang, 
            config=self.model.config_string
        )
        return OCRResult(text=text.strip(), confidence=1.0)

    def _recognize_batch(self, images: list[np.ndarray]) -> list[OCRResult]:
        return [self._recognize_single(img) for img in images]

    def recognize(
        self, 
        image: np.ndarray | list[np.ndarray]
    ) -> OCRResult | list[OCRResult]:
        
        if isinstance(image, list):
            return self._recognize_batch(image)
        return self._recognize_single(image)
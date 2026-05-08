# -- built in
from dataclasses import dataclass

# -- 3rd party
import numpy as np
import pytesseract

# -- own
from .ocr import BaseOCR, OCRResult, TextBlock

@dataclass
class TesseractParams:
    language: str = 'vie'
    page_segmentation_mode: int = 3
    orc_engine_mode: int = 3

class TesseractModel:
    def __init__(self, params: TesseractParams | None = None):
        # configured params or defaults
        self.params = params or TesseractParams()

        self.config_string = f"--oem {self.params.orc_engine_mode} --psm {self.params.page_segmentation_mode}"

class TesseractOCR(BaseOCR):
    def __init__(self, model: TesseractModel | None = None):
        super().__init__()
        self.model = model or TesseractModel()

    def _recognize_single(self, image: np.ndarray) -> OCRResult:
        data = pytesseract.image_to_data(
            image, 
            lang    = self.model.params.language, 
            config  = self.model.config_string,
            output_type = pytesseract.Output.DICT
        )

        return self._parse_data_result(data)

    def _parse_data_result(self, data: dict) -> OCRResult:
        texts = []
        
        rows = zip(
            data['text'], 
            data['conf'], 
            data['left'], 
            data['top'], 
            data['width'], 
            data['height']
        )

        for text, conf, left, top, width, height in rows:
            if text.strip() == "":
                continue

            if text:
                confidence = float(conf) / 100.0 if conf != '-1' else 0.0
                text_block = TextBlock(
                    text = text.strip(),

                    bounding_polygon = [
                        (left, top), (left + width, top),
                        (left + width, top + height), (left, top + height)
                    ],
                    confidence = confidence
                )

                texts.append(text_block)

        return OCRResult(texts = texts)

    def _recognize_batch(self, images: list[np.ndarray]) -> list[OCRResult]:
        return [self._recognize_single(img) for img in images]

    def recognize(
            self, 
            image: np.ndarray | list[np.ndarray]
        ) -> OCRResult | list[OCRResult]:
        
        if isinstance(image, list):
            return self._recognize_batch(image)
        return self._recognize_single(image)
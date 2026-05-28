from __future__     import annotations

import numpy as np

from .adapter       import OCRAdapter
from .confidence    import ConfidenceScorer
from .models        import OCRPageResult
# from .paddle import PaddleTextRecognizer
from .tesseract     import TesseractOCR


class OCRPipeline:
    def __init__(self, engine: str = "tesseract", threshold: float = 0.8) -> None:
        self.adapter = OCRAdapter()
        self.scorer = ConfidenceScorer(threshold=threshold)

        if engine == "tesseract":
            self.engine = TesseractOCR()
        # elif engine == "paddle":
        #     self.engine = PaddleTextRecognizer()
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def run(self, image: np.ndarray) -> OCRPageResult:
        block_result = self.engine.recognize(image)
        page_result = self.adapter.convert(block_result)
        return self.scorer.score(page_result)
from dataclasses import dataclass
from typing import List, Optional, ForwardRef

OCRResult = ForwardRef('OCRResult')


@dataclass
class Correction:
    original_text: str
    corrected_text: str
    confidence: float
    position: int
    reason: str


@dataclass
class CorrectionResult:
    corrected_texts: List[str]
    original_texts: List[str]
    corrections: List[Correction]
    corrected_count: int = 0
    ocr_result: Optional['OCRResult'] = None
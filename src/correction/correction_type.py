from dataclasses import dataclass
from typing import List, Optional, ForwardRef

OCRResult = ForwardRef('OCRResult')


@dataclass
class Correction:
    """Lưu thông tin một lần sửa lỗi"""
    original_text: str
    corrected_text: str
    confidence: float          # confidence trước khi sửa
    position: int              # index trong list blocks
    reason: str                # ví dụ: "rule", "dictionary", "model"


@dataclass
class CorrectionResult:
    """Kết quả sau khi sửa lỗi"""
    ocr_result: 'OCRResult'           
    corrections: List[Correction]
    corrected_count: int = 0
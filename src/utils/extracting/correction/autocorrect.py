from typing import List, Union, Optional
from .correction_type import Correction, CorrectionResult
from .rule import RuleCorrector
from .dictionary import DictionaryCorrector


class AutoCorrector:
    """
    Module Auto-correction độc lập cho VN-Digitize
    Có thể dùng riêng lẻ hoặc tích hợp sau
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.min_confidence = self.config.get('min_confidence', 0.75)
        
        self.rule_corrector = RuleCorrector()
        self.dict_corrector = DictionaryCorrector(confidence_threshold=self.min_confidence)

    def correct_text(self, text: str) -> str:
        """Sửa lỗi cho 1 đoạn text duy nhất"""
        if not self.enabled or not text:
            return text
        
        texts = [text]
        result = self.correct_list(texts)

        return result.corrected_texts[0]

    def correct_list(self, texts: List[str], confidences: List[float] = None) -> CorrectionResult:
        """Sửa lỗi cho danh sách text"""
        if not self.enabled:
            return CorrectionResult(
                corrected_texts=texts,
                original_texts=texts,
                corrections=[],
                corrected_count=0
            )

        current_texts = texts.copy()
        all_corrections = []

        # Rule-based
        current_texts, rule_corrections = self.rule_corrector.correct(current_texts)
        all_corrections.extend(rule_corrections)

        # Dictionary-based
        current_texts, dict_corrections = self.dict_corrector.correct(current_texts, confidences)
        all_corrections.extend(dict_corrections)

        return CorrectionResult(
            corrected_texts=current_texts,
            original_texts=texts,
            corrections=all_corrections,
            corrected_count=len(all_corrections)
        )

    def correct_ocr_result(self, ocr_result) -> CorrectionResult:
        """Hỗ trợ OCRResult"""
        confidences = None
        if hasattr(ocr_result, 'texts'):
            if isinstance(ocr_result.texts, str):
                texts = [ocr_result.texts]
            else:
                texts = [block.text for block in ocr_result.texts]
                confidences = [block.confidence for block in ocr_result.texts]
        else:
            texts = [str(ocr_result)]

        result = self.correct_list(texts, confidences)
        result.ocr_result = ocr_result  # lưu tham chiếu gốc
        return result
        
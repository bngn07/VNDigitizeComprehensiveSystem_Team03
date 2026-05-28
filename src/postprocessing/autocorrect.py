from __future__     import annotations

from dataclasses    import dataclass, field
from typing         import Any

from .rules         import RuleCorrector
from .types         import CorrectionResult


@dataclass(slots=True)
class AutoCorrector:

    enabled: bool = True
    min_confidence: float = 0.75
    rule_corrector: RuleCorrector = field(default_factory = RuleCorrector)
    dictionary_corrector: Any | None = None

    def correct_text(self, text: str) -> str:
        if not self.enabled or not text:
            return text

        result = self.correct_list([text])
        return result.corrected_texts[0]

    def correct_list(
        self,
        texts: list[str],
        confidences: list[float] | None = None,
    ) -> CorrectionResult:
        if not self.enabled:
            return CorrectionResult(
                corrected_texts = list(texts),
                original_texts  = list(texts),
                corrections     = [],
                corrected_count = 0
            )

        current_texts = list(texts)
        all_corrections = []

        current_texts, rule_corrections = self.rule_corrector.correct(current_texts)
        all_corrections.extend(rule_corrections)

        if self.dictionary_corrector is not None:
            current_texts, dictionary_corrections = self.dictionary_corrector.correct(
                current_texts,
                confidences,
            )
            all_corrections.extend(dictionary_corrections)

        return CorrectionResult(
            corrected_texts = current_texts,
            original_texts  = list(texts),
            corrections     = all_corrections,
            corrected_count = len(all_corrections)
        )

    def correct_ocr_result(self, ocr_result: Any) -> CorrectionResult:
        texts, confidences = self._extract_texts_and_confidences(ocr_result)
        result = self.correct_list(texts, confidences)
        result.ocr_result = ocr_result
        return result

    @staticmethod
    def _extract_texts_and_confidences(
        ocr_result: Any,
    ) -> tuple[list[str], list[float] | None]:
        if not hasattr(ocr_result, "texts"):
            return [str(ocr_result)], None

        texts_attr = ocr_result.texts
        if isinstance(texts_attr, str):
            return [texts_attr], None

        texts = [block.text for block in texts_attr]
        confidences = [block.confidence for block in texts_attr]
        return texts, confidences

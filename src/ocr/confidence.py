from __future__ import annotations

from .models import OCRPageResult


class ConfidenceScorer:
    def __init__(self, threshold: float = 0.8) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], received {threshold}")
        self.threshold = threshold

    def score(self, result: OCRPageResult) -> OCRPageResult:
        if not result.words:
            result.overall_confidence = 0.0
            return result

        total_confidence = 0.0
        for word in result.words:
            word.flagged = word.confidence < self.threshold
            total_confidence += word.confidence

        result.overall_confidence = total_confidence / len(result.words)
        return result

    def get_flagged(self, result: OCRPageResult) -> list:
        return [word for word in result.words if word.flagged]

    def summary(self, result: OCRPageResult) -> dict:
        flagged = self.get_flagged(result)
        return {
            "total_words": len(result.words),
            "flagged_count": len(flagged),
            "flagged_words": [word.text for word in flagged],
            "overall_confidence": round(result.overall_confidence * 100, 2),
            "threshold": self.threshold * 100,
        }

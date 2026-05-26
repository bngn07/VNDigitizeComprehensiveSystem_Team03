from .schema import WordResult, OCRResult

# ====================================================================
# CONFIDENCE SCORER
# --------------------------------------------------------------------
class ConfidenceScorer:
    def __init__(self, threshold: float = 0.8):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], received {threshold}")
        self.threshold = threshold

    def score(self, result: OCRResult) -> OCRResult:
        """
        Duyệt qua từng từ, flag nếu confidence < threshold,
        sau đó tính overall_confidence cho cả trang.

        Args:
            result: OCRResult chứa danh sách các WordResult
        Return:
            OCRResult đã được cập nhật flagged và overall_confidence
        """
        if not result.words:
            return result

        total_conf = 0.0
        for word in result.words:
            word.flagged = word.confidence < self.threshold
            total_conf += word.confidence

        result.overall_confidence = total_conf / len(result.words)
        return result

    def get_flagged(self, result: OCRResult):
        """Trả về danh sách các từ bị flag"""
        return [w for w in result.words if w.flagged]

    def summary(self, result: OCRResult) -> dict:
        """Tóm tắt kết quả confidence của cả trang"""
        flagged = self.get_flagged(result)
        return {
            "total_words": len(result.words),
            "flagged_count": len(flagged),
            "flagged_words": [w.text for w in flagged],
            "overall_confidence": round(result.overall_confidence * 100, 2),
            "threshold": self.threshold * 100
        }
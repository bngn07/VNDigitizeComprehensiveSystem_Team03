
#   Convert OCRResult của (TextBlock) sang WordResult
# ====================================================================
from .ocr import OCRResult as GroupOCRResult
from .schema import WordResult, OCRResult as MyOCRResult


class OCRAdapter:
    def convert(self, group_result: GroupOCRResult) -> MyOCRResult:
        """
        Nhận OCRResult
        Trả về OCRResult (có WordResult, flagged, v.v.)
        """
        # Trường hợp trả về raw string, không có TextBlock
        if isinstance(group_result.texts, str):
            return MyOCRResult(
                words=[],
                raw_text=group_result.texts,
                overall_confidence=0.0
            )

        words = []
        for block in group_result.texts:
            # Lấy bounding box từ TextBlock
            x, y, width, height = block.bounding_box()

            # Tách từng từ trong block (1 block có thể có nhiều từ)
            word_list = block.text.strip().split()
            word_width = width // len(word_list) if word_list else width

            for i, word in enumerate(word_list):
                words.append(WordResult(
                    text=word,
                    confidence=round(block.confidence, 4),
                    x=x + i * word_width,
                    y=y,
                    width=word_width,
                    height=height
                ))

        return MyOCRResult(words=words)
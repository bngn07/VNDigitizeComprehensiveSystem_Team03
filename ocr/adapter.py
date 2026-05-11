
from .ocr import OCRResult as GroupOCRResult
from .schema import WordResult, OCRResult as MyOCRResult


class OCRAdapter:
    def convert(self, group_result: GroupOCRResult) -> MyOCRResult:
        """
        Nhận OCRResult từ nhóm (Task 1)
        Trả về OCRResult của Task 3 (có WordResult, flagged, v.v.)
        """
        # Trường hợp nhóm trả về raw string
        if isinstance(group_result.texts, str):
            return MyOCRResult(
                words=[],
                raw_text=group_result.texts,
                overall_confidence=0.0
            )

        # Trường hợp list rỗng
        if not group_result.texts:
            return MyOCRResult(words=[])

        words = []
        for block in group_result.texts:
            # Lấy bounding box từ TextBlock
            x, y, width, height = block.bounding_box()

            # Tách từng từ trong block
            word_list = block.text.strip().split()
            if not word_list:
                continue

            word_width = width // len(word_list)

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
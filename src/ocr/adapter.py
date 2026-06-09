from __future__ import annotations

from .models    import (OCRBlockResult, 
                        OCRPageResult, 
                        WordResult)


class OCRAdapter:
    def convert(self, block_result: OCRBlockResult) -> OCRPageResult:
        if isinstance(block_result.blocks, str):
            return OCRPageResult(
                words = [],
                raw_text = block_result.blocks,
                overall_confidence = 0.0,
            )

        if not block_result.blocks:
            return OCRPageResult(words=[])

        words: list[WordResult] = []
        for block in block_result.blocks:
            x, y, width, height = block.bounding_box()
            split_words = block.text.strip().split()
            if not split_words:
                continue

            word_width = max(1, width // len(split_words))
            for index, word in enumerate(split_words):
                words.append(
                    WordResult(
                        text        = word,
                        confidence  = round(block.confidence, 4),
                        x           = x + index * word_width,
                        y           = y,
                        width       = word_width,
                        height      = height,
                    )
                )

        return OCRPageResult(
            words = words,
            overall_confidence = block_result.confidence,
        )

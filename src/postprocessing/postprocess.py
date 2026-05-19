from dataclasses import dataclass, field
from typing import Union

from ocr.ocr import OCRResult, TextBlock
from src.utils.extractor import DocumentExtractor

# @dataclass
# class PostprocessResult:
#     full_text: str
#     blocks: list[TextBlock]
#     metadata: dict

class Postprocessing:  
    def __init__(self, config: dict = None):
        self.config = config if config is not None else {}
        self.extractor = DocumentExtractor()

    def _merge_lines(
            self, 
            texts: list[TextBlock] | str, 
            y_threshold_ratio: float = 0.5) -> OCRResult:
        
        if not texts or isinstance(texts, str):
            return OCRResult(texts=texts)

        blocks_info = []
        for block in texts:
            min_x, min_y, w, h = block.bounding_box()
            center_y = min_y + (h / 2)
            blocks_info.append({
                'block': block,
                'center_y': center_y,
                'min_x': min_x,
                'min_y': min_y,
                'max_x': min_x + w,
                'max_y': min_y + h,
                'height': h
            })

        blocks_info.sort(key=lambda b: b['center_y'])

        lines = []
        current_line = [blocks_info[0]]

        for current_block in blocks_info[1:]:
            anchor_block = current_line[0]
            y_diff = abs(current_block['center_y'] - anchor_block['center_y'])

            if y_diff < (anchor_block['height'] * y_threshold_ratio):
                current_line.append(current_block)
            else:
                lines.append(current_line)
                current_line = [current_block]

        if current_line:
            lines.append(current_line)

        merged_blocks = []
        for line in lines:
            line.sort(key=lambda b: b['min_x'])
            merged_text = " ".join([b['block'].text for b in line])
            
            merged_conf = sum(b['block'].confidence for b in line) / len(line)
            
            line_min_x = min(b['min_x'] for b in line)
            line_min_y = min(b['min_y'] for b in line)
            line_max_x = max(b['max_x'] for b in line)
            line_max_y = max(b['max_y'] for b in line)
            
            merged_polygon = [
                (line_min_x, line_min_y),
                (line_max_x, line_min_y),
                (line_max_x, line_max_y),
                (line_min_x, line_max_y)
            ]

            merged_blocks.append(TextBlock(
                text = merged_text,
                bounding_polygon = merged_polygon,
                confidence = merged_conf
            ))

        return OCRResult(texts = merged_blocks)

    def process(self, raw_blocks: OCRResult) -> OCRResult:
        return self._merge_lines(raw_blocks.texts)

    def process_and_extract(self, raw_blocks: OCRResult, words: list = None) -> dict:
        
        merged = self._merge_lines(raw_blocks.texts)

        if isinstance(merged.texts, str):
            raw_text = merged.texts
        else:
            raw_text = " ".join(b.text for b in merged.texts)

        fields = self.extractor.extract(raw_text, words=words)

        return {
            "ocr_result": merged,
            "fields":     fields,
        }
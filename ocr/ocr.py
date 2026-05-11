import numpy as np
import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import overload

@dataclass(frozen=True)
class TextBlock:
    text: str
    bounding_polygon: list[tuple[int, int]]
    confidence: float

    def bounding_box(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.bounding_polygon]
        ys = [p[1] for p in self.bounding_polygon]
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)

    def __str__(self) -> str:
        display_text = self.text if len(self.text) <= 40 else self.text[:37] + "..."
        box = self.bounding_box()
        
        return f'[Conf: {self.confidence:.4f}] "{display_text}" | Box: {box}'


@dataclass(frozen=True)
class OCRResult:
    texts: list[TextBlock] | str

    @property
    def confidence(self) -> float:
        if not self.texts or isinstance(self.texts, str):
            return 0.0
        return sum(block.confidence for block in self.texts) / len(self.texts)

    def to_json(self, output_path: str = None) -> str:
        data_dict = asdict(self)
        data_dict['confidence'] = self.confidence
        json_string = json.dumps(data_dict, ensure_ascii=False)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_string)
            print(f"output -> {output_path}")

        return json_string

    def __str__(self) -> str:
        lines = ["=" * 50]
        lines.append(" OCR INFERENCE RESULT ")
        lines.append("=" * 50)
        
        if isinstance(self.texts, str):
            lines.append(" Output Type  : Raw String")
            lines.append(" Confidence   : N/A")
            lines.append("-" * 50)
            lines.append(f" {self.texts}")
            
        else:
            lines.append(f" Total Blocks : {len(self.texts)}")
            lines.append(f" Avg Conf     : {self.confidence:.4f}")
            lines.append("-" * 50)
            
            for i, block in enumerate(self.texts):
                lines.append(f" [{i:02d}] {block}")
                
        lines.append("=" * 50)
        return "\n".join(lines)


class BaseOCR(ABC):
    @overload
    def recognize(self, image: np.ndarray) -> OCRResult: ...
    
    @overload
    def recognize(self, image: list[np.ndarray]) -> list[OCRResult]: ...

    @abstractmethod
    def recognize(
            self, 
            image: np.ndarray | list[np.ndarray]
        ) -> OCRResult | list[OCRResult]:
        pass
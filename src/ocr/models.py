from __future__     import annotations

from dataclasses    import asdict, dataclass, field
from typing         import Any


@dataclass
class TextBlock:
    text: str
    polygon: list[tuple[int, int]]
    confidence: float

    def bounding_box(self, fmt: str = "ltwh") -> tuple[int, int, int, int]:
        if not self.polygon:
            return 0, 0, 0, 0

        xs = [point[0] for point in self.polygon]
        ys = [point[1] for point in self.polygon]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        if fmt == "xyxy":
            return min_x, min_y, max_x, max_y
        return min_x, min_y, max_x - min_x, max_y - min_y


@dataclass
class OCRBlockResult:
    blocks: list[TextBlock] | str

    @property
    def confidence(self) -> float:
        if isinstance(self.blocks, str):
            return 0.0
        if not self.blocks:
            return 0.0
        values = [block.confidence for block in self.blocks]
        return sum(values) / len(values)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"] = self.confidence
        return data


@dataclass
class WordResult:
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    flagged: bool = False


@dataclass
class OCRPageResult:
    words: list[WordResult] = field(default_factory=list)
    raw_text: str = ""
    overall_confidence: float = 0.0
    
    def __post_init__(self) -> None:
        if not self.raw_text and self.words:
            self.raw_text = " ".join(word.text for word in self.words)
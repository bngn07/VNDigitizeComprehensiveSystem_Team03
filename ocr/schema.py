
from dataclasses import dataclass, field
from typing import List

# ====================================================================
# OCR PARAMS
# ====================================================================
@dataclass
class OCRParams:
    lang: str = "vie"
    confidence_threshold: float = 0.8
    engine: str = "tesseract"

# ====================================================================
# WORD RESULT
# ====================================================================
@dataclass
class WordResult:
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    flagged: bool = False

    def __str__(self):
        if self.flagged:
            status = "FLAG"
        else:
            status = "OK"

        result = (
            f"[{status}] '{self.text}' "
            f"| confidence: {self.confidence*100:.1f}% "
            f"| bbox: ({self.x}, {self.y}, {self.width}, {self.height})"
        )
        return result

# ====================================================================
# OCR RESULT
# ====================================================================
@dataclass
class OCRResult:
    words: List[WordResult] = field(default_factory=list)
    raw_text: str = ""
    overall_confidence: float = 0.0

    def __post_init__(self):
        if not self.raw_text:
            if self.words:
                word_texts = []
                for w in self.words:
                    word_texts.append(w.text)
                self.raw_text = " ".join(word_texts)

    def __str__(self):
        parts = []
        parts.append("OCR RESULT")
        parts.append(f"  raw_text           : {self.raw_text}")
        parts.append(f"  overall_confidence : {self.overall_confidence*100:.1f}%")
        parts.append(f"  total_words        : {len(self.words)}")

        flagged = []
        for w in self.words:
            if w.flagged:
                flagged.append(w)

        parts.append(f"  flagged_words      : {len(flagged)}")

        if len(flagged) > 0:
            parts.append("\nFLAGGED WORDS")
            for w in flagged:
                parts.append(f"  {w}")

        return "\n".join(parts)
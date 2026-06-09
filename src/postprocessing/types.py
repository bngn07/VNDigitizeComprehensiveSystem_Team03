from __future__     import annotations

from dataclasses    import dataclass
from typing         import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


@dataclass(slots = True)
class Correction:
    original_text: str
    corrected_text: str
    confidence: float
    position: int
    reason: str


@dataclass(slots=True)
class CorrectionResult:
    corrected_texts: list[str]
    original_texts: list[str]
    corrections: list[Correction]
    corrected_count: int = 0
    ocr_result: Any | None = None
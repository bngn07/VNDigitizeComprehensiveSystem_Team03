from __future__     import annotations

from dataclasses    import dataclass, field
from typing         import Any

import numpy as np


@dataclass
class DecisionResult:
    label: int
    label_name: str
    confidence: float
    recipe: list[str]
    probs: dict[str, float]


@dataclass
class CodeResult:
    type: str
    content: str
    bbox: tuple[int, int, int, int]
    polygon: list[tuple[int, int]] | None = None
    quality: int | None = None
    orientation: str | None = None
    raw_bytes: bytes | None = None


@dataclass
class RotationResult:
    angle: float
    confidence: float
    script: str = "Unknown"


@dataclass
class PreprocessResult:
    image: np.ndarray
    metadata: dict[str, Any]
    decision: DecisionResult | None = field(default=None, repr=False)

    def __str__(self) -> str:
        parts: list[str] = []

        if self.image is None:
            parts.append("IMAGE None")
        else:
            height, width = self.image.shape[:2]
            channels = self.image.shape[2] if self.image.ndim == 3 else 1
            parts.append(
                f"IMAGE size={width}x{height} channels={channels} dtype={self.image.dtype}"
            )

        if self.decision is None:
            parts.append("DECISION none")
        else:
            probs = ", ".join(
                f"{key}={value:.3f}" for key, value in self.decision.probs.items()
            )
            parts.append(
                "DECISION "
                f"label={self.decision.label_name} "
                f"confidence={self.decision.confidence:.4f} "
                f"recipe={self.decision.recipe} "
                f"probs={probs}"
            )

        status = self.metadata.get("status")
        qrcodes = self.metadata.get("qrcodes", [])
        parts.append(f"META status={status} qrcount={len(qrcodes)}")

        return " | ".join(parts)

from typing import Any
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol


class CodeDetector:
    def __init__(self, types: list[str] = None):
        if types is not None:
            valid = {s.name for s in ZBarSymbol}
            invalid = set(types) - valid
            if invalid:
                raise ValueError(
                    f"Unknown code types: {invalid}. Valid types: {sorted(valid)}"
                )
        self._types = types

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        if image is None or image.size == 0:
            return []

        if self._types:
            symbols = [ZBarSymbol[t] for t in self._types]
            results = decode(image, symbols=symbols)
        else:
            results = decode(image)

        return [self._parse(r) for r in results]

    def _parse(self, r) -> dict[str, Any]:
        return {
            "type": r.type,
            "content": r.data.decode("utf-8", errors="replace"),
            "bbox": (r.rect.left, r.rect.top, r.rect.width, r.rect.height),
            "polygon": [(p.x, p.y) for p in r.polygon],
            "quality": r.quality,
        }
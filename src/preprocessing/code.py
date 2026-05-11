from typing import Any
from dataclasses import dataclass

import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol

@dataclass
class CodeResult:
    type: str
    content: str
    bbox: tuple[int, int, int, int]
    polygon: list[tuple[int, int]]
    quality: int

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

    def detect(self, image: np.ndarray) -> list[CodeResult]: # can be more than one code in a page
        if image is None or image.size == 0:
            return []

        if self._types:
            symbols = [ZBarSymbol[t] for t in self._types]
            results = decode(image, symbols=symbols)
        else:
            results = decode(image)

        return [self._parse(r) for r in results]
    
    # result: Any because in the source code of pyzbar, the return type of decode is not well defined. IIt can return weird objects that are not documented according to the author of the lib himself.
    def _parse(self, result: Any) -> CodeResult:
        # return {
        #     "type": result.type,
        #     "content": result.data.decode("utf-8", errors="replace"),
        #     "bbox": (result.rect.left, result.rect.top, result.rect.width, result.rect.height),
        #     "polygon": [(p.x, p.y) for p in result.polygon],
        #     "quality": result.quality,
        # }

        return CodeResult(
            type=result.type, 
            content=result.data.decode("utf-8", errors="replace"),
            bbox=(result.rect.left, result.rect.top, result.rect.width, result.rect.height),
            polygon=[(p.x, p.y) for p in result.polygon],
            quality=result.quality   
        )
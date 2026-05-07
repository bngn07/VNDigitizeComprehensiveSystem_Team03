import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import overload

@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float

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
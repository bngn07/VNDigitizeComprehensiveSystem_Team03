from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .models import OCRBlockResult


class BaseOCR(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray | list[np.ndarray]) -> OCRBlockResult | list[OCRBlockResult]:
        raise NotImplementedError

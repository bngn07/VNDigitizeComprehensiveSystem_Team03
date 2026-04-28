from abc import ABC, abstractmethod
from typing import Any, Tuple

class IBlankDetector(ABC):
    @abstractmethod
    def is_blank(self, image: Any) -> Tuple[bool, float, str]:
        pass
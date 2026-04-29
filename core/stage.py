import logging
from abc import ABC, abstractmethod
from .data_models import OCRContext

logger = logging.getLogger(__name__)

class OCRStage(ABC):
    name: str = "UnnamedStage"

    @abstractmethod
    def process(self, ctx: OCRContext) -> OCRContext:
        pass
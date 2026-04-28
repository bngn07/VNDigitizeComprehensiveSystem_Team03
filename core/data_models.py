from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

@dataclass
class PageMetadata:
    is_blank: bool = False;
    blank_score: float = 0.0;
    blank_reason: str = "untested"

class DocumentPage:
    page_number: int
    original_image: np.ndarray
    metadata: PageMetadata = field(default_factory = PageMetadata)
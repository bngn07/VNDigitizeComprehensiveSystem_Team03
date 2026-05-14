from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias, overload

from .schema import CourtRecord

InputText: TypeAlias = str

@dataclass(frozen=True)
class Prompt:
    text: str

    def __str__(self) -> str:
        return self.text

@dataclass
class ExtractedResult:

    record: CourtRecord

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict (``None`` for absent fields)."""
        return self.record.model_dump()

    def __str__(self) -> str:
        parts = ["EXTRACTION RESULT"]
        for k, v in self.to_dict().items():
            parts.append(f"  {k:<12}: {v}")
        return "\n".join(parts)

class BaseExtractor(ABC):
    @abstractmethod
    def _build_prompt(self, input_text: InputText) -> Prompt: pass

    @abstractmethod
    def _extract_single(self, input_text: InputText) -> ExtractedResult: pass
    
    def _extract_batch(
        self, inputs: list[InputText]
    ) -> list[ExtractedResult]:
        return [self._extract_single(inp) for inp in inputs]
    
    @overload
    def extract(self, input_text: InputText) -> ExtractedResult: ...

    @overload
    def extract(self, input_text: list[InputText]) -> list[ExtractedResult]: ...

    def extract(
        self,
        input_text: InputText | list[InputText],
    ) -> ExtractedResult | list[ExtractedResult]:
        if isinstance(input_text, list):
            for i, item in enumerate(input_text):
                if not isinstance(item, str):
                    raise TypeError(
                        f"All inputs must be str; item at index {i} is "
                        f"{type(item).__name__}"
                    )
            return self._extract_batch(input_text)

        if not isinstance(input_text, str):
            raise TypeError(
                f"input_text must be a str or list[str], "
                f"got {type(input_text).__name__}"
            )
        return self._extract_single(input_text)

# -- built in --
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import overload, Generic, TypeVar, TypeAlias, Optional, Type, Any, Dict

# -- third party --
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

T = TypeVar("T", bound=BaseModel)
TextInput: TypeAlias = str | list[str]


# ── DEFINE ─────────────────────────────────────────
@dataclass
class ExtractorResult(Generic[T]):
    record: Optional[T] = None
    schema_used: str = ""
    extracted_dynamic_data: Optional[Dict[str, Any]] = None
    confidence_overall: float = 0.0
    fill_rate: float = 0.0
    error: Optional[str] = None
    raw_output_sample: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_used": self.schema_used,
            "confidence_overall": self.confidence_overall,
            "fill_rate": self.fill_rate,
            "error": self.error,
            "extracted_dynamic_data": self.extracted_dynamic_data,
            "record": self.record.model_dump() if self.record else None
        }

    def __str__(self) -> str:
        parts = [f"EXTRACTION RESULT (Schema: {self.schema_used} | Conf: {self.confidence_overall:.1%} | Fill: {self.fill_rate:.1%})"]
        if self.error:
            parts.append(f"  [ERROR]: {self.error}")
        if self.record:
            for k, v in self.record.model_dump().items():
                parts.append(f"  {k:<25}: {v}")
        return "\n".join(parts)


class BaseExtractor(ABC):
    @overload
    def extract(self, inputs: str, schema: Optional[Type[T]] = None) -> ExtractorResult[T]: ...

    @overload
    def extract(self, inputs: list[str], schema: Optional[Type[T]] = None) -> list[ExtractorResult[T]]: ...

    def extract(
        self,
        inputs: TextInput,
        schema: Optional[Type[T]] = None,
    ) -> ExtractorResult[T] | list[ExtractorResult[T]]:
        if isinstance(inputs, list):
            return self._extract_batch(inputs, schema)
        return self._extract_single(inputs, schema)

    @abstractmethod
    def _extract_single(self, text: str, schema: Optional[Type[T]] = None) -> ExtractorResult[T]:
        pass

    def _extract_batch(
        self, texts: list[str], schema: Optional[Type[T]] = None
    ) -> list[ExtractorResult[T]]:
        # ML/DL models with native batch support should override this method.
        with ThreadPoolExecutor() as executor:
            return list(executor.map(lambda t: self._extract_single(t, schema), texts))
# ── END ─────────────────────────────────────────
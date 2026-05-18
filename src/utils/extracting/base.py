# -- built in --
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import overload, Generic, TypeVar, TypeAlias

# -- third party --
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

T = TypeVar("T", bound=BaseModel)
TextInput: TypeAlias = str | list[str]


# ── DEFINE ─────────────────────────────────────────
@dataclass
class ExtractorResult(Generic[T]):
    record: T

    def to_dict(self) -> dict[str, object]:
        return self.record.model_dump()

    def __str__(self) -> str:
        parts = ["EXTRACTION RESULT"]
        for k, v in self.to_dict().items():
            parts.append(f"  {k:<12}: {v}")
        return "\n".join(parts)

class BaseExtractor(ABC):
    @overload
    def extract(
        self,
        inputs: TextInput,
        schema: type[T],
    ) -> ExtractorResult[T]: ...

    @overload
    def extract(
        self,
        inputs: list[TextInput],
        schema: type[T],
    ) -> list[ExtractorResult[T]]: ...

    def extract(
        self,
        inputs: TextInput | list[TextInput],
        schema: type[T],
    ) -> ExtractorResult[T] | list[ExtractorResult[T]]:
        if isinstance(inputs, list):
            return self._extract_batch(inputs, schema)
        return self._extract_single(inputs, schema)

    @abstractmethod
    def _extract_single(self, text: TextInput, schema: type[T]) -> ExtractorResult[T]:
        pass

    def _extract_batch(
        self, texts: list[TextInput], schema: type[T]
    ) -> list[ExtractorResult[T]]:
        # ML/DL models with native batch support should override this method.
        with ThreadPoolExecutor() as executor:
            return list(executor.map(lambda t: self._extract_single(t, schema), texts))
# ── END ─────────────────────────────────────────
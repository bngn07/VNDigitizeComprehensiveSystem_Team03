from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, overload

from pydantic import BaseModel

T = TypeVar("T", bound = BaseModel)
TextInput = str | list[str]


@dataclass(slots=True)
class ExtractorResult(Generic[T]):
    record: T | None = None
    schema_used: str = ""
    extracted_dynamic_data: dict[str, Any] | None = None
    confidence_overall: float = 0.0
    fill_rate: float = 0.0
    error: str | None = None
    raw_output_sample: str = ""

    def to_dict(self, document_id: str = "") -> dict[str, Any]:
        metadata = self.build_metadata_list()
        return {
            "document_id": document_id,
            "classification": self.schema_used,
            "confidence_overall": self.confidence_overall,
            "fill_rate": self.fill_rate,
            "error": self.error,
            "metadata": metadata,
        }

    def build_metadata_list(self) -> list[dict[str, Any]]:
        if not self.extracted_dynamic_data:
            if self.record is None:
                return []

            return [
                {
                    "field_name": key,
                    "value": value,
                    "confidence": None,
                    "flagged": False,
                    "missing": value is None,
                }
                for key, value in self.record.model_dump().items()
            ]

        metadata: list[dict[str, Any]] = []
        for field_name, field_info in self.extracted_dynamic_data.items():
            if isinstance(field_info, dict):
                entry = {
                    "field_name": field_name,
                    "value": field_info.get("value"),
                    "confidence": field_info.get("confidence"),
                    "flagged": field_info.get("flagged", False),
                    "missing": field_info.get(
                        "missing",
                        field_info.get("value") is None,
                    ),
                }
                if "warning" in field_info:
                    entry["warning"] = field_info["warning"]
            else:
                entry = {
                    "field_name": field_name,
                    "value": field_info,
                    "confidence": None,
                    "flagged": False,
                    "missing": field_info is None,
                }
            metadata.append(entry)

        return metadata


class BaseExtractor(ABC):
    """Base interface for sync and batched extractors."""

    @overload
    def extract(
        self,
        inputs: str,
        schema: type[T] | None = None,
    ) -> ExtractorResult[T]:
        ...

    @overload
    def extract(
        self,
        inputs: list[str],
        schema: type[T] | None = None,
    ) -> list[ExtractorResult[T]]:
        ...

    def extract(
        self,
        inputs: TextInput,
        schema: type[T] | None = None,
    ) -> ExtractorResult[T] | list[ExtractorResult[T]]:
        if isinstance(inputs, list):
            return self._extract_batch(inputs, schema)
        return self._extract_single(inputs, schema)

    @abstractmethod
    def _extract_single(
        self,
        text: str,
        schema: type[T] | None = None,
    ) -> ExtractorResult[T]:
        raise NotImplementedError

    def _extract_batch(
        self,
        texts: list[str],
        schema: type[T] | None = None,
    ) -> list[ExtractorResult[T]]:
        with ThreadPoolExecutor() as executor:
            return list(executor.map(lambda item: self._extract_single(item, schema), texts))

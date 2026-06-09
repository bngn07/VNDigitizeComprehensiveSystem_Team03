from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import overload, Generic, TypeVar, TypeAlias, Optional, Type, Any, Dict, List

from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

T = TypeVar("T", bound=BaseModel)
TextInput: TypeAlias = str | list[str]



@dataclass
class ExtractorResult(Generic[T]):
    record: Optional[T] = None
    schema_used: str = ""
    extracted_dynamic_data: Optional[Dict[str, Any]] = None
    confidence_overall: float = 0.0
    fill_rate: float = 0.0
    error: Optional[str] = None
    raw_output_sample: str = ""

    def to_dict(self, document_id: str = "") -> dict:
        metadata = self._build_metadata_list()
        return {
            "document_id":        document_id,
            "classification":     self.schema_used,
            "confidence_overall": self.confidence_overall,
            "fill_rate":          self.fill_rate,
            "error":              self.error,
            "metadata":           metadata,
        }

    def _build_metadata_list(self) -> List[dict]:
        if not self.extracted_dynamic_data:
            if self.record:
                return [
                    {
                        "field_name":  k,
                        "value":       v,
                        "confidence":  None,
                        "flagged":     False,
                        "missing":     v is None,
                    }
                    for k, v in self.record.model_dump().items()
                ]
            return []

        metadata: List[dict] = []
        for field_name, field_info in self.extracted_dynamic_data.items():
            if isinstance(field_info, dict):
                entry = {
                    "field_name":  field_name,
                    "value":       field_info.get("value"),
                    "confidence":  field_info.get("confidence"),
                    "flagged":     field_info.get("flagged", False),
                    "missing":     field_info.get("missing", field_info.get("value") is None),
                }
                if "warning" in field_info:
                    entry["warning"] = field_info["warning"]
            else:
                entry = {
                    "field_name":  field_name,
                    "value":       field_info,
                    "confidence":  None,
                    "flagged":     False,
                    "missing":     field_info is None,
                }
            metadata.append(entry)
        return metadata

    def to_flat_dict(self) -> dict:
        return {
            "schema_used":            self.schema_used,
            "confidence_overall":     self.confidence_overall,
            "fill_rate":              self.fill_rate,
            "error":                  self.error,
            "extracted_dynamic_data": self.extracted_dynamic_data,
            "record":                 self.record.model_dump() if self.record else None,
        }

    def __str__(self) -> str:
        parts = [
            f"EXTRACTION RESULT "
            f"(Schema: {self.schema_used} | "
            f"Conf: {self.confidence_overall:.1%} | "
            f"Fill: {self.fill_rate:.1%})"
        ]
        if self.error:
            parts.append(f"  [ERROR]: {self.error}")
        metadata = self._build_metadata_list()
        for m in metadata:
            conf_str  = f"{m['confidence']:.2f}" if m["confidence"] is not None else "N/A"
            flag_str  = " ⚠️"  if m.get("flagged") else ""
            warn_str  = f"  ← {m['warning']}" if m.get("warning") else ""
            parts.append(
                f"  {m['field_name']:<30}: {str(m['value'])!r:<50} "
                f"(conf={conf_str}){flag_str}{warn_str}"
            )
        return "\n".join(parts)



class BaseExtractor(ABC):
    @overload
    def extract(self, inputs: str,       schema: Optional[Type[T]] = None) -> ExtractorResult[T]: ...
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
    def _extract_single(
        self, text: str, schema: Optional[Type[T]] = None
    ) -> ExtractorResult[T]:
        pass

    def _extract_batch(
        self, texts: list[str], schema: Optional[Type[T]] = None
    ) -> list[ExtractorResult[T]]:
        with ThreadPoolExecutor() as executor:
            return list(
                executor.map(lambda t: self._extract_single(t, schema), texts)
            )
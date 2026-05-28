from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from json_repair import repair_json
from pydantic import BaseModel

from ..base import BaseExtractor, ExtractorResult, T
from ..prompt import Prompt
from ..schema import DonTuCamKetSchema, SCHEMA_REGISTRY, auto_detect_schema
from ..scoring import score_extraction

logger = logging.getLogger(__name__)


@dataclass(slots = True)
class GeminiParams:
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 512
    max_input_chars: int = 3000
    max_input_chars_retry: int = 1500
    max_retries: int = 2
    retry_delay: float = 1.5
    low_confidence_threshold: float = 0.65


class GeminiExtractor(BaseExtractor):
    def __init__(self, api_key: str, params: GeminiParams | None = None) -> None:
        self.params = params or GeminiParams()
        self.client = genai.Client(api_key=api_key)
        self.generate_config = types.GenerateContentConfig(
            response_mime_type = "application/json",
            max_output_tokens = self.params.max_output_tokens,
            temperature = self.params.temperature,
        )

    def _extract_single(
        self,
        text: str,
        schema: type[T] | None = None,
    ) -> ExtractorResult[T]:
        document_id = f"doc_{uuid.uuid4().hex[:8]}"
        schema_name, schema_class = self._resolve_schema(text, schema)
        last_raw = ""

        for attempt in range(1, self.params.max_retries + 2):
            try:
                prompt = Prompt.build(
                    schema_class    = schema_class,
                    raw_text        = text,
                    retry           = attempt > 1,
                    max_chars       = self.params.max_input_chars,
                    max_chars_retry = self.params.max_input_chars_retry,
                )
                response = self.client.models.generate_content(
                    model = self.params.model_name,
                    contents = str(prompt),
                    config = self.generate_config,
                )
                last_raw = response.text or ""

                if self._clean_and_parse(last_raw, document_id):
                    return self._build_result(
                        document_id  = document_id,
                        schema_name  = schema_name,
                        schema_class = schema_class,
                        raw_output   = last_raw,
                    )
            except Exception as exc:
                logger.exception("Gemini extraction failed on attempt %s: %s", attempt, exc)

            time.sleep(self.params.retry_delay * attempt)

        return ExtractorResult(
            schema_used = schema_name,
            error = "failed_after_retries",
            raw_output_sample = last_raw[:300],
        )

    def _resolve_schema(
        self,
        text: str,
        schema: type[T] | None,
    ) -> tuple[str, type[BaseModel]]:
        if schema is not None:
            return schema.__name__, schema

        schema_name = auto_detect_schema(text)
        return schema_name, SCHEMA_REGISTRY.get(schema_name, DonTuCamKetSchema)

    def _build_result(
        self,
        document_id: str,
        schema_name: str,
        schema_class: type[BaseModel],
        raw_output: str,
    ) -> ExtractorResult:
        parsed = self._clean_and_parse(raw_output, document_id)
        if not parsed:
            return ExtractorResult(
                schema_used = schema_name,
                error = "json_parse_failed",
                raw_output_sample = raw_output[:300],
            )

        metadata_list, overall, fill_rate = score_extraction(
            parsed,
            schema_class,
            self.params.low_confidence_threshold,
        )
        record_data = {item["field_name"]: item["value"] for item in metadata_list}

        try:
            record = schema_class(**record_data)
        except Exception:
            record = None

        return ExtractorResult(
            record      = record,
            schema_used = schema_name,
            extracted_dynamic_data = {item["field_name"]: item for item in metadata_list},
            confidence_overall = overall,
            fill_rate   = fill_rate,
            raw_output_sample = raw_output[:300],
        )

    @staticmethod
    def _clean_and_parse(raw: str, document_id: str = "") -> dict[str, Any] | None:
        raw = raw.strip()
        markdown_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        candidate = markdown_match.group(1).strip() if markdown_match else raw

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1:
            candidate = candidate[start : end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        try:
            repaired = repair_json(candidate, return_objects=True)
            if isinstance(repaired, dict) and repaired:
                return repaired
        except Exception:
            logger.warning("JSON repair failed for %s", document_id)

        return None
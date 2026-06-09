from __future__ import annotations

from dataclasses import dataclass

import ollama

from ..base import BaseExtractor, ExtractorResult, T
from ..prompt import Prompt


@dataclass(slots = True)
class GemmaParams:
    model_name: str = "gemma4:e2b"
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64


class GemmaExtractor(BaseExtractor):
    def __init__(self, params: GemmaParams | None = None) -> None:
        self.params = params or GemmaParams()
        self.client = ollama.Client()

    def _extract_single(
        self,
        text: str,
        schema: type[T] | None = None,
    ) -> ExtractorResult[T]:
        if schema is None:
            raise ValueError("GemmaExtractor requires an explicit schema.")

        prompt = Prompt.build(schema_class = schema, raw_text = text)
        response = self.client.chat(
            model = self.params.model_name,
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a structured data extractor. "
                        "Return only valid JSON matching the provided schema."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt.text,
                },
            ],
            format = schema.model_json_schema(),
            options = {
                "temperature": self.params.temperature,
                "top_p": self.params.top_p,
                "top_k": self.params.top_k,
            },
        )

        raw_content = response["message"]["content"].strip()
        record = schema.model_validate_json(raw_content)
        return ExtractorResult(record=record, schema_used=schema.__name__)

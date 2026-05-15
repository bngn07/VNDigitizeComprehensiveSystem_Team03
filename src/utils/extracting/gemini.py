# -- built in --
import os
from dataclasses import dataclass

# -- third party --
from google import genai
from google.genai import types

# -- self-defined --
from .base import (
        ExtractorResult, 
        BaseExtractor, 
        TextInput, T)
from .prompt import Prompt


@dataclass
class GeminiParams:
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.1


class GeminiExtractor(BaseExtractor):

    def __init__(self, params: GeminiParams | None = None, **kwargs) -> None:
        self.params = params or GeminiParams()
        self._client: object = None
        self._types: object = None

        self._get_client()
        super().__init__()

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client

        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. "
                "Pass GOOGLE_API_KEY to OS environment."
            )

        self._client = genai.Client(api_key=key)
        self._types = types
        return self._client

    def _build_prompt(self, input_text: TextInput) -> Prompt:
        return Prompt.build(ocr_text = input_text)

    def _extract_single(self, input_text: TextInput, schema: type[T]) -> ExtractorResult[T]:
        client = self._client
        prompt = self._build_prompt(input_text)

        response = client.models.generate_content(
            model=self.params.model_name,
            contents=prompt.text,
            config=self._types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=self.params.temperature,
            ),
        )
        
        record = schema.model_validate_json(response.text.strip())
        return ExtractorResult(record = record)
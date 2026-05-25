# -- built in --
from dataclasses import dataclass

# -- third party --
import ollama
from pydantic import BaseModel

# -- self-defined --
from .base import (
    ExtractorResult, 
    BaseExtractor, 
    TextInput, T
)
from .prompt import Prompt


@dataclass
class Gemma4Params:
    model_name: str = "gemma4:e2b" 
    
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64


class GemmaExtractor(BaseExtractor):

    def __init__(self, params: Gemma4Params | None = None, **kwargs) -> None:
        self.params = params or Gemma4Params()
        self._client: ollama.Client | None = None

        self._get_client()
        super().__init__()

    def _get_client(self) -> ollama.Client:
        if self._client is not None:
            return self._client
        
        self._client = ollama.Client()
        return self._client

    def _build_prompt(self, input_text: TextInput) -> Prompt:
        return Prompt.build(ocr_text=input_text)

    def _extract_single(self, input_text: TextInput, schema: type[T]) -> ExtractorResult[T]:
        client = self._client
        prompt = self._build_prompt(input_text)

        system_content = (
            "You are a highly capable data extraction assistant. "
            "You must extract information and output ONLY valid JSON matching the exact schema provided."
        )

        response = client.chat(
            model=self.params.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": prompt.text
                }
            ],
            format=schema.model_json_schema(),
            options={
                "temperature": self.params.temperature,
                "top_p": self.params.top_p,
                "top_k": self.params.top_k,
            }
        )
        
        raw_content = response['message']['content'].strip()
        
        try:
            record = schema.model_validate_json(raw_content)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON. Raw output: {raw_content}\nError: {e}")

        return ExtractorResult(record=record)
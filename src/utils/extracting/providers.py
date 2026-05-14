from __future__ import annotations

import os
from typing import Any

from .base   import BaseExtractor, ExtractedResult, InputText, Prompt
from .schema import CourtRecord


# ====================================================================
# PROMPT TEMPLATES
# ====================================================================

_SYSTEM_PROMPT = """\
Bạn là một trợ lý trích xuất thông tin từ văn bản OCR của tài liệu pháp lý Việt Nam.
Nhiệm vụ của bạn là trích xuất chính xác các trường sau từ văn bản được cung cấp và trả về \
dưới dạng JSON hợp lệ.

Các trường cần trích xuất:
- so_ban_an  : Số bản án (ví dụ: "746/2017/HS-PT"). Nếu không tìm thấy, trả về null.
- ten_bi_cao : Tên bị cáo (ví dụ: "Đỗ Văn N"). Nếu không tìm thấy, trả về null.
- toi_danh   : Tội danh bị truy tố (ví dụ: "Tội trộm cắp tài sản"). Nếu không tìm thấy, trả về null.
- nam_sinh   : Năm sinh của bị cáo dưới dạng số nguyên (ví dụ: 1948). Nếu không tìm thấy, trả về null.

Chỉ trả về JSON, không giải thích thêm.\
"""

_USER_TEMPLATE = """\
Văn bản OCR:
{ocr_text}
"""


# ====================================================================
# GEMINI EXTRACTOR  (Google Generative AI — new google.genai SDK)
# ====================================================================

class GeminiExtractor(BaseExtractor):
    """
    Calls the Google Generative AI API to perform structured extraction.

    Uses the new ``google.genai`` SDK with ``response_schema=CourtRecord``
    so the model returns validated JSON directly.  The underlying client
    is initialised lazily on the first call to :meth:`_extract_single`
    so that importing or instantiating this class is safe even when
    ``GOOGLE_API_KEY`` is not set.

    Parameters
    ----------
    api_key : str | None
        Explicit API key.  Falls back to the ``GOOGLE_API_KEY``
        environment variable when *None*.
    model : str
        Gemini model identifier (e.g. ``"gemini-2.5-flash"``).
    temperature : float
        Sampling temperature.  Keep low (≈ 0.1) for deterministic
        field extraction.

    Example
    -------
    ::

        ext = GeminiExtractor(api_key="AIza...")
        result = ext.extract("Số bản án: 746/2017/HS-PT ...")
        print(result.to_dict())
    """

    DEFAULT_MODEL       = "gemini-2.5-flash"
    DEFAULT_TEMPERATURE = 0.1

    def __init__(
        self,
        api_key:     str | None = None,
        model:       str        = DEFAULT_MODEL,
        temperature: float      = DEFAULT_TEMPERATURE,
    ) -> None:
        self._api_key     = api_key
        self.model        = model
        self.temperature  = temperature
        self._client: Any = None   # lazily initialised
        self._types:  Any = None   # lazily imported google.genai.types

    # ----------------------------------------------------------------
    # Lazy client initialisation
    # ----------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return (and create, if needed) the genai client."""
        if self._client is not None:
            return self._client

        try:
            from google import genai          # noqa: PLC0415
            from google.genai import types    # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiExtractor.  "
                "Install it with:  pip install google-genai"
            ) from exc

        key = self._api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set.  "
                "Pass api_key= to GeminiExtractor or export GOOGLE_API_KEY."
            )

        self._client = genai.Client(api_key=key)
        self._types  = types
        return self._client

    # ----------------------------------------------------------------
    # BaseExtractor interface
    # ----------------------------------------------------------------

    def _build_prompt(self, input_text: InputText) -> Prompt:
        """Assemble the full extraction prompt for *input_text*."""
        text = _SYSTEM_PROMPT + "\n\n" + _USER_TEMPLATE.format(ocr_text=input_text)
        return Prompt(text=text)

    def _extract_single(self, input_text: InputText) -> ExtractedResult:
        """Call Gemini API and return a structured :class:`ExtractedResult`."""
        client = self._get_client()
        prompt = self._build_prompt(input_text)

        response = client.models.generate_content(
            model=self.model,
            contents=prompt.text,
            config=self._types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CourtRecord,
                temperature=self.temperature,
            ),
        )
        raw_json = response.text.strip()
        record   = CourtRecord.model_validate_json(raw_json)
        return ExtractedResult(record=record)


# ====================================================================
# LOCAL EXTRACTOR  (Ollama-compatible / extensible)
# ====================================================================

class LocalExtractor(BaseExtractor):
    """
    Adapter for local / self-hosted LLMs (e.g. Ollama, llama.cpp,
    vLLM, or any OpenAI-compatible server).

    The default implementation targets the Ollama ``/api/generate``
    endpoint.  Subclass and override :meth:`_call_model` for other
    servers.

    Gemma models are supported by passing the model name directly,
    e.g. ``model='gemma3'`` or ``model='gemma4'`` depending on which
    version is available in your local Ollama installation.

    Parameters
    ----------
    base_url : str
        Base URL of the local inference server
        (e.g. ``"http://localhost:11434/api/generate"``).
    model : str
        Model identifier understood by the server.  Examples:
        ``"llama3"``, ``"gemma3"``, ``"gemma4"``.
    temperature : float
        Sampling temperature.
    timeout : int
        HTTP request timeout in seconds.

    Example
    -------
    ::

        # Ollama default
        ext = LocalExtractor()
        result = ext.extract("Số bản án: 746/2017/HS-PT ...")

        # Gemma via Ollama
        ext = LocalExtractor(model="gemma3")
        result = ext.extract("...")
    """

    DEFAULT_BASE_URL    = "http://localhost:11434/api/generate"
    DEFAULT_MODEL       = "llama3"
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_TIMEOUT     = 60

    def __init__(
        self,
        base_url:    str   = DEFAULT_BASE_URL,
        model:       str   = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout:     int   = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url    = base_url
        self.model       = model
        self.temperature = temperature
        self.timeout     = timeout

    # ----------------------------------------------------------------
    # Internal model call (override for different backends)
    # ----------------------------------------------------------------

    def _call_model(self, prompt: str) -> str:
        """
        Send *prompt* to the local server; return raw response text.

        Default implementation targets an Ollama-style
        ``/api/generate`` endpoint.  Override for other servers.
        """
        try:
            import requests   # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "requests is required for LocalExtractor.  "
                "Install it with:  pip install requests"
            ) from exc

        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        resp = requests.post(
            self.base_url,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns {"response": "..."}
        return data.get("response", data.get("text", str(data)))

    # ----------------------------------------------------------------
    # BaseExtractor interface
    # ----------------------------------------------------------------

    def _build_prompt(self, input_text: InputText) -> Prompt:
        """Assemble the full extraction prompt for *input_text*."""
        text = _SYSTEM_PROMPT + "\n\n" + _USER_TEMPLATE.format(ocr_text=input_text)
        return Prompt(text=text)

    def _extract_single(self, input_text: InputText) -> ExtractedResult:
        """Call the local model and return a structured :class:`ExtractedResult`."""
        prompt   = self._build_prompt(input_text)
        raw_json = self._call_model(prompt.text).strip()
        record   = CourtRecord.model_validate_json(raw_json)
        return ExtractedResult(record=record)


# ====================================================================
# BACKWARD-COMPATIBILITY ALIASES
# ====================================================================
# Older code that imported BaseExtractorProvider, GeminiProvider, or
# LocalProvider continues to work without modification.

BaseExtractorProvider = BaseExtractor   # type: ignore[assignment,misc]
GeminiProvider        = GeminiExtractor
LocalProvider         = LocalExtractor

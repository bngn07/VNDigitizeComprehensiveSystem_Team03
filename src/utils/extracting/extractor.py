from __future__ import annotations

from typing import overload

from .base      import BaseExtractor, ExtractedResult, InputText
from .config    import ExtractorConfig, PROVIDER_GEMINI, PROVIDER_LOCAL
from .providers import GeminiExtractor, LocalExtractor
from .schema    import CourtRecord

ExtractionResult = ExtractedResult

class Extractor:
    
    def __init__(self, config: ExtractorConfig | dict | None = None) -> None:
        if config is None:
            self.config: ExtractorConfig = ExtractorConfig()
        elif isinstance(config, dict):
            self.config = ExtractorConfig.from_dict(config)
        elif isinstance(config, ExtractorConfig):
            self.config = config
        else:
            raise TypeError(
                f"config must be an ExtractorConfig, dict, or None; "
                f"got {type(config).__name__}"
            )

        self._provider: BaseExtractor | None = None   # lazy

    # ----------------------------------------------------------------
    # Provider resolution
    # ----------------------------------------------------------------

    def _resolve_provider(self) -> BaseExtractor:
        """Instantiate and return the configured backend extractor."""
        cfg   = self.config
        extra = dict(cfg.provider_kwargs)

        if cfg.model is not None:
            extra.setdefault("model", cfg.model)

        extra.setdefault("temperature", cfg.temperature)

        if cfg.provider == PROVIDER_GEMINI:
            if cfg.api_key is not None:
                extra.setdefault("api_key", cfg.api_key)
            return GeminiExtractor(**extra)

        if cfg.provider == PROVIDER_LOCAL:
            return LocalExtractor(**extra)

        raise ValueError(
            f"Unknown provider '{cfg.provider}'.  "
            f"Valid choices: {PROVIDER_GEMINI!r}, {PROVIDER_LOCAL!r}"
        )

    def _get_provider(self) -> BaseExtractor:
        if self._provider is None:
            self._provider = self._resolve_provider()
        return self._provider

    # ----------------------------------------------------------------
    # Public API — overloaded dispatch
    # ----------------------------------------------------------------

    @overload
    def extract(self, ocr_text: str) -> ExtractedResult: ...

    @overload
    def extract(self, ocr_text: list[str]) -> list[ExtractedResult]: ...

    def extract(
        self,
        ocr_text: str | list[str],
    ) -> ExtractedResult | list[ExtractedResult]:
        """
        Parse *ocr_text* and return an :class:`ExtractedResult` (single)
        or a ``list[ExtractedResult]`` (batch).

        Missing fields are represented as ``None`` in the result dict.

        Parameters
        ----------
        ocr_text : str | list[str]
            Raw string(s) produced by an OCR engine.

        Returns
        -------
        ExtractedResult
            When *ocr_text* is a single string.
        list[ExtractedResult]
            When *ocr_text* is a list of strings.

        Raises
        ------
        TypeError
            When *ocr_text* is not a ``str`` or ``list[str]``.
        EnvironmentError
            When the API key is missing and the provider requires one.
        ImportError
            When a required optional library (e.g. ``google-genai``)
            is not installed.
        """
        if isinstance(ocr_text, list):
            for i, item in enumerate(ocr_text):
                if not isinstance(item, str):
                    raise TypeError(
                        f"All inputs must be str; item at index {i} is "
                        f"{type(item).__name__}"
                    )
            provider = self._get_provider()
            try:
                return provider._extract_batch(ocr_text)
            except Exception as exc:
                print(f"ERR [Extractor] provider raised an exception: {exc}")
                raise

        if not isinstance(ocr_text, str):
            raise TypeError(
                f"ocr_text must be a str or list[str], "
                f"got {type(ocr_text).__name__}"
            )

        provider = self._get_provider()
        try:
            return provider._extract_single(ocr_text)
        except Exception as exc:
            print(f"ERR [Extractor] provider raised an exception: {exc}")
            raise

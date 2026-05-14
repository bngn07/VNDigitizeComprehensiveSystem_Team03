from __future__ import annotations

from dataclasses import dataclass, field

PROVIDER_GEMINI = "gemini"
PROVIDER_LOCAL  = "local"

PROVIDER_NAMES: dict[str, str] = {
    PROVIDER_GEMINI: "GeminiProvider",
    PROVIDER_LOCAL:  "LocalProvider",
}

@dataclass
class ExtractorConfig:
    provider:         str         = PROVIDER_GEMINI
    api_key:          str | None  = None
    model:            str | None  = None
    temperature:      float       = 0.1
    provider_kwargs:  dict        = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict) -> "ExtractorConfig":
        return cls(
            provider        = cfg.get("provider",         PROVIDER_GEMINI),
            api_key         = cfg.get("api_key",          None),
            model           = cfg.get("model",            None),
            temperature     = cfg.get("temperature",      0.1),
            provider_kwargs = cfg.get("provider_kwargs",  {}),
        )

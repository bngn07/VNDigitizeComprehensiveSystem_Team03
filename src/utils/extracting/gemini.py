# -- built in --
import os
import re
import json
import time
import math
import logging
from dataclasses import dataclass
from typing import Optional, Type, Tuple, Any, List

# -- third party --
from google import genai
from google.genai import types
from json_repair import repair_json
from tqdm.auto import tqdm

# -- self-defined --
from .base import ExtractorResult, BaseExtractor, TextInput, T
from .prompt import Prompt
from .schema import auto_detect_schema

logger = logging.getLogger(__name__)

@dataclass
class GeminiParams:
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.0
    max_output_tokens: int = 512
    max_input_chars: int = 3000
    max_input_chars_retry: int = 1500
    batch_size: int = 10
    batch_poll_interval: int = 20
    batch_timeout_sec: int = 7200
    max_retries: int = 2
    retry_delay: float = 1.5
    low_confidence_threshold: float = 0.65


# ── UTILS (JSON Parser & Confidence Scorer) ──────────────────────
def clean_and_parse(raw: str, doc_id: str = "") -> Optional[dict]:
    raw = raw.strip()
    m = re.search(r"http://googleusercontent.com/immersive_entry_chip/0", raw)


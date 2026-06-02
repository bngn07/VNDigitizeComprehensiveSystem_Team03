from __future__  import annotations

import re
from dataclasses import dataclass, field

from .types      import Correction


@dataclass(slots=True)
class RuleCorrector:
    rules: list[tuple[re.Pattern[str], str]] = field(
        default_factory = lambda: [
            (re.compile(r"l(\d)"), r"1\1"),
            (re.compile(r"(\d)l"), r"\g<1>1"),
            (re.compile(r"O(\d)"), r"0\1"),
            (re.compile(r"(\d)O"), r"\g<1>0"),
            (re.compile(r"\s+"), " "),
            (re.compile(r"\s*-\s*"), "-"),
        ]
    )

    def correct(self, texts: list[str]) -> tuple[list[str], list[Correction]]:
        corrected_texts = list(texts)
        corrections: list[Correction] = []

        for index, text in enumerate(texts):
            current_text = text

            for pattern, replacement in self.rules:
                updated_text = pattern.sub(replacement, current_text)
                if updated_text == current_text:
                    continue

                corrections.append(
                    Correction(
                        original_text   = current_text,
                        corrected_text  = updated_text,
                        confidence      = 1.0,
                        position        = index,
                        reason          = "rule-based",
                    )
                )
                current_text = updated_text

            corrected_texts[index] = current_text

        return corrected_texts, corrections
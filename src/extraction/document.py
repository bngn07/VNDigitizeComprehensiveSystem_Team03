from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any


class DocumentExtractor:
    def extract(self, raw_text: str, words: list[Any] | None = None) -> dict[str, Any]:
        issuing_authority = self._extract_authority(raw_text)

        return {
            "metadata": [
                self._make_field("ten_loai_van_ban", self._extract_document_type(raw_text), words),
                self._make_field("so_van_ban", self._extract_document_number(raw_text), words),
                self._make_field("ky_hieu", self._extract_symbol(raw_text), words),
                self._make_field("ngay_thang_nam", self._extract_date(raw_text), words),
                self._make_field("co_quan_ban_hanh", issuing_authority, words),
            ]
        }

    def _make_field(
        self,
        field_name: str,
        value: str | None,
        words: list[Any] | None,
    ) -> dict[str, Any]:
        return {
            "field_name": field_name,
            "value": value or None,
            "confidence": self._calc_confidence(value, words) if value else 0.0,
            "bounding_box": None,
        }

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().replace("đ", "d").replace("Đ", "d")
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(char for char in normalized if not unicodedata.combining(char))

    @staticmethod
    def _calc_confidence(value: str | None, words: list[Any] | None) -> float:
        if not value or not words:
            return 0.0
        return 0.8

    def _extract_symbol(self, text: str) -> str:
        header = text[:400]
        patterns = [
            r"[Ss][ôo][t]?\s*:\s*[0-9]+/[0-9]+/([A-ZĐa-zđ\-]+)",
            r"[Ss][ôo][t]?\s*:\s*[0-9]+/([A-ZĐa-zđ\-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, header)
            if match:
                return match.group(1).strip(".,;: ")
        return ""

    def _extract_date(self, text: str) -> str:
        patterns = [
            r"[Nn]gày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
            r"[Nn]gày\s*:\s*(\d{1,2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{4})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                day, month, year = match.groups()
                return f"{int(day):02d}/{int(month):02d}/{year}"
        return ""

    def _extract_document_type(self, text: str) -> str:
        header = text[:300]
        normalized = self._normalize(header)
        keywords = {
            "quyet dinh": "Quyết định",
            "ban an": "Bản án",
            "bien ban": "Biên bản",
            "cong van": "Công văn",
            "hop dong": "Hợp đồng",
            "don": "Đơn",
        }
        for key, value in keywords.items():
            if key in normalized:
                return value

        candidates = list(keywords.keys())
        close_match = difflib.get_close_matches(normalized[:50], candidates, n=1, cutoff=0.6)
        return keywords[close_match[0]] if close_match else ""

    @staticmethod
    def _extract_document_number(text: str) -> str:
        header = text[:400]
        patterns = [
            r"[Ss][ốo][t]?\s*:\s*([0-9]+\s*/[^\s,;:]+)",
            r"[Ss][ốo][t]?\s+([0-9]+/[^\s,;:]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, header, re.MULTILINE)
            if match:
                return match.group(1).strip(".,;: ").replace(" ", "")
        return ""

    @staticmethod
    def _extract_authority(text: str) -> str:
        patterns = [
            r"(TÒA ÁN NHÂN DÂN(?:\s+\S+){1,4})",
            r"((?:ỦY BAN|UỶ BAN) NHÂN DÂN(?:\s+\S+){1,4})",
            r"(VIỆN KIỂM SÁT NHÂN DÂN(?:\s+\S+){1,4})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip(".,;:")
        return ""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel

KEY_PATTERNS = [
    re.compile(r"(?:bị đơn|nguyên đơn|bị cáo)[:\s].+", re.IGNORECASE),
    re.compile(r"\d{1,3}/\d{4}/[A-Z]{2,}"),
    re.compile(r"ngày\s+\d{1,2}\s+tháng", re.IGNORECASE),
    re.compile(r"(?:thẩm phán|kiểm sát viên|điều tra viên)[:\s].+", re.IGNORECASE),
]

SYSTEM_HEADER = (
    "Bạn là chuyên gia trích xuất dữ liệu văn bản pháp lý Việt Nam.\n"
    "Nhiệm vụ: đọc văn bản và trả về JSON với các trường cho sẵn.\n"
    "Quy tắc QUAN TRỌNG:\n"
    "1. Chỉ trả về JSON thuần tuý — không markdown, không giải thích.\n"
    "2. Không tìm thấy → null.\n"
    "3. Nhiều người → nối bằng '; '.\n"
    "4. Giữ nguyên key, chỉ điền value.\n"
    "5. Luôn thêm key 'summary' với tóm tắt 2–3 câu tiếng Việt.\n"
)


def smart_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    cut = int(max_chars * 0.78)
    head = text[:cut]
    tail = text[cut:]
    key_lines = [
        line.strip()
        for line in tail.splitlines()
        if line.strip() and any(pattern.search(line) for pattern in KEY_PATTERNS)
    ]
    snippet = " | ".join(key_lines)[: int(max_chars * 0.22)]
    return head + ("\n...[cắt]...\n" + snippet if snippet else " ...[cắt]")


def build_prompt(
    schema_class: type[BaseModel],
    raw_text: str,
    retry: bool = False,
    max_chars: int = 3000,
    max_chars_retry: int = 1500,
) -> str:
    max_length = max_chars_retry if retry else max_chars
    text = smart_truncate(raw_text, max_length)

    if retry:
        keys = list(schema_class.model_json_schema().get("properties", {}).keys())
        return (
            f"Trích xuất {', '.join(keys)} từ văn bản pháp lý tiếng Việt.\n"
            "Trả về JSON thuần tuý. Không tìm thấy → null.\n\n"
            f"VĂN BẢN:\n{text}"
        )

    return (
        f"{SYSTEM_HEADER}"
        f"\nTRƯỜNG CẦN TRÍCH XUẤT:\n{format_field_list(schema_class)}\n\n"
        f"TEMPLATE:\n{build_empty_template(schema_class)}\n\n"
        f"VĂN BẢN:\n{text}"
    )


def format_field_list(schema_class: type[BaseModel]) -> str:
    props = schema_class.model_json_schema().get("properties", {})
    return "\n".join(
        f"- {field_name}: {details.get('description', '')}"
        for field_name, details in props.items()
    )


def build_empty_template(schema_class: type[BaseModel]) -> str:
    props = schema_class.model_json_schema().get("properties", {})
    template = {key: None for key in props}
    template["summary"] = None
    return json.dumps(template, ensure_ascii=False, indent=2)


@dataclass(frozen=True, slots=True)
class Prompt:
    text: str

    @classmethod
    def build(
        cls,
        schema_class: type[BaseModel],
        raw_text: str,
        retry: bool = False,
        max_chars: int = 3000,
        max_chars_retry: int = 1500,
    ) -> "Prompt":
        return cls(
            text=build_prompt(
                schema_class=schema_class,
                raw_text=raw_text,
                retry=retry,
                max_chars=max_chars,
                max_chars_retry=max_chars_retry,
            )
        )

    def __str__(self) -> str:
        return self.text

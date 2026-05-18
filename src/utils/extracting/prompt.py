import re
import json
from dataclasses import dataclass
from typing import Type
from pydantic import BaseModel

_KEY_PATS = [
    re.compile(r"(?:bị đơn|nguyên đơn|bị cáo)[:\s].+", re.IGNORECASE),
    re.compile(r"\d{1,3}/\d{4}/[A-Z]{2,}"),
    re.compile(r"ngày\s+\d{1,2}\s+tháng", re.IGNORECASE),
    re.compile(r"(?:thẩm phán|kiểm sát viên|điều tra viên)[:\s].+", re.IGNORECASE),
]

def smart_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = int(max_chars * 0.78)
    head = text[:cut]
    tail = text[cut:]
    key_lines = [
        ln.strip() for ln in tail.splitlines()
        if ln.strip() and any(p.search(ln) for p in _KEY_PATS)
    ]
    snippet = " | ".join(key_lines)[:int(max_chars * 0.22)]
    return head + ("\n...[cắt]...\n" + snippet if snippet else " ...[cắt]")

def _field_list(sc: Type[BaseModel]) -> str:
    props = sc.model_json_schema().get("properties", {})
    return "\n".join(f"- {k}: {v.get('description','')}" for k, v in props.items())

def _empty_template(sc: Type[BaseModel]) -> str:
    props = sc.model_json_schema().get("properties", {})
    return json.dumps({k: None for k in props}, ensure_ascii=False, indent=2)

_SYSTEM_HEADER = (
    "Bạn là chuyên gia trích xuất dữ liệu văn bản pháp lý Việt Nam.\n"
    "Nhiệm vụ: đọc văn bản và trả về JSON với các trường cho sẵn.\n"
    "Quy tắc QUAN TRỌNG:\n"
    "1. Chỉ trả về JSON thuần tuý — không markdown, không giải thích.\n"
    "2. Không tìm thấy → null (không dùng chuỗi rỗng).\n"
    "3. Nhiều người → nối bằng '; '.\n"
    "4. Giữ nguyên key, chỉ điền value.\n"
)

@dataclass(frozen=True)
class Prompt:
    text: str

    @classmethod
    def build(cls, schema_class: Type[BaseModel], raw_text: str, retry: bool = False, max_chars: int = 3000, max_chars_retry: int = 1500) -> "Prompt":
        max_c = max_chars_retry if retry else max_chars
        text  = smart_truncate(raw_text, max_c)
        
        if retry:
            keys = list(schema_class.model_json_schema().get("properties", {}).keys())
            prompt_text = (
                f"Trích xuất {', '.join(keys)} từ văn bản pháp lý tiếng Việt.\n"
                "Trả về JSON. Không tìm thấy → null.\n\n"
                f"VĂN BẢN:\n{text}"
            )
        else:
            prompt_text = (
                f"{_SYSTEM_HEADER}"
                f"\nTRƯỜNG CẦN TRÍCH XUẤT:\n{_field_list(schema_class)}\n\n"
                f"TEMPLATE:\n{_empty_template(schema_class)}\n\n"
                f"VĂN BẢN:\n{text}"
            )
        return cls(text=prompt_text)

    def __str__(self) -> str:
        return self.text
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
    template = {k: None for k in props}
    template["summary"] = None  # thêm field summary
    return json.dumps(template, ensure_ascii=False, indent=2)

_SYSTEM_HEADER = (
    "Bạn là chuyên gia trích xuất dữ liệu văn bản pháp lý Việt Nam.\n"
    "Nhiệm vụ: đọc văn bản và trả về JSON với các trường cho sẵn.\n"
    "Quy tắc QUAN TRỌNG:\n"
    "1. Chỉ trả về JSON thuần tuý — không markdown, không giải thích.\n"
    "2. Không tìm thấy → null (không dùng chuỗi rỗng).\n"
    "3. Nhiều người → nối bằng '; '.\n"
    "4. Giữ nguyên key, chỉ điền value.\n"
    "5. Luôn thêm key 'summary': tóm tắt nội dung văn bản trong 2–3 câu tiếng Việt.\n"
)


def build_prompt(
    schema_class: Type[BaseModel],
    raw_text: str,
    retry: bool = False,
    max_chars: int = 3000,
    max_chars_retry: int = 1500,
) -> str:
    """
    Thay thế Prompt.build() — trả về str thay vì Prompt object.
    Thêm multipage hint và content_page_hint khi retry.
    """
    max_c = max_chars_retry if retry else max_chars
    text  = smart_truncate(raw_text, max_c)

    is_multipage = bool(
        re.search(r"\n{3,}|={5,}|─{5,}|\[trang\s*\d+\]|page\s*\d+", text, re.IGNORECASE)
    )
    multipage_hint = (
        "\nLƯU Ý: Văn bản gồm nhiều trang — thông tin như số hiệu, ngày, tên cơ quan "
        "có thể xuất hiện ở trang sau, không nhất thiết ở đầu văn bản. "
        "Hãy đọc toàn bộ và trích xuất dù thông tin nằm ở bất kỳ trang nào.\n"
        if is_multipage else ""
    )

    if retry:
        keys = list(schema_class.model_json_schema().get("properties", {}).keys())
        content_page_hint = (
            "\nVăn bản này có thể là trang nội dung (thiếu tiêu đề/số hiệu ở đầu). "
            "Hãy suy luận từ các dấu hiệu trong nội dung: danh sách đương sự, "
            "tên tranh chấp, cơ quan được nhắc đến... để điền các trường còn thiếu. "
            "Nếu thực sự không có thông tin → null.\n"
        )
        return (
            f"Trích xuất {', '.join(keys)} từ văn bản pháp lý tiếng Việt.\n"
            f"{multipage_hint}"
            f"{content_page_hint}"
            "Trả về JSON thuần tuý. Không tìm thấy → null.\n\n"
            f"VĂN BẢN:\n{text}"
        )

    return (
        f"{_SYSTEM_HEADER}"
        f"{multipage_hint}"
        f"\nTRƯỜNG CẦN TRÍCH XUẤT:\n{_field_list(schema_class)}\n\n"
        f"TEMPLATE:\n{_empty_template(schema_class)}\n\n"
        f"VĂN BẢN:\n{text}"
    )


# Giữ lại class Prompt để tương thích ngược với code cũ dùng Prompt.build()
@dataclass(frozen=True)
class Prompt:
    text: str

    @classmethod
    def build(
        cls,
        schema_class: Type[BaseModel],
        raw_text: str,
        retry: bool = False,
        max_chars: int = 3000,
        max_chars_retry: int = 1500,
    ) -> "Prompt":
        return cls(text=build_prompt(schema_class, raw_text, retry, max_chars, max_chars_retry))

    def __str__(self) -> str:
        return self.text
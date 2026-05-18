from dataclasses import dataclass


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


@dataclass(frozen=True)
class Prompt:
    text: str

    @classmethod
    def build(cls, ocr_text: str) -> "Prompt":
        text = _SYSTEM_PROMPT + "\n\n" + _USER_TEMPLATE.format(ocr_text=ocr_text)
        return cls(text=text)

    def __str__(self) -> str:
        return self.text
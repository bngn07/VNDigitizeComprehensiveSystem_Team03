"""
Trích xuất metadata từ nội dung văn bản:
    - ten_loai_vb  : nhãn phân loại
    - so_van_ban   : số hiệu văn bản (VD: 12/2024/QĐ-PT)
    - ngay_ky      : ngày ký (DD/MM/YYYY)
    - co_quan_ban_hanh : tên cơ quan ký ban hành
"""
import re

from config.settings import RE_SO_VAN_BAN


_RE_DATE = re.compile(
    r'ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})'
    r'|\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b',
    re.I,
)
_RE_AGENCY = re.compile(
    r'(TÒA ÁN[\w\s]+|UBND[\w\s,]+|BỘ[\w\s]+'
    r'|CỤC[\w\s]+|SỞ[\w\s]+|VIỆN KIỂM SÁT[\w\s]+)',
    re.I,
)


def extract_metadata(pages: list[dict], label: str, confidence: float) -> list[dict]:
    """
    Trích xuất metadata từ tối đa 2 trang đầu của segment.

    Trả về list[dict] với cấu trúc:
        [{"field_name": str, "value": str, "confidence": float}, ...]
    """
    head = " ".join(p["text_full"] for p in pages[:2])[:3000]
    meta = [{"field_name": "ten_loai_vb", "value": label, "confidence": round(confidence, 3)}]

    # Số văn bản
    found_ids = RE_SO_VAN_BAN.findall(head)
    if found_ids:
        meta.append({"field_name": "so_van_ban", "value": found_ids[0].upper(), "confidence": 0.97})

    # Ngày ký
    m = _RE_DATE.search(head)
    if m:
        g = m.groups()
        day, month, year = (g[0], g[1], g[2]) if g[0] else (g[3], g[4], g[5])
        meta.append({
            "field_name": "ngay_ky",
            "value"     : f"{day.zfill(2)}/{month.zfill(2)}/{year}",
            "confidence": 0.92,
        })

    # Cơ quan ban hành — ưu tiên match theo dòng trước, fallback regex
    agency = None
    for line in head.split("\n"):
        s = line.strip()
        if len(s) >= 5 and _RE_AGENCY.match(s):
            agency = s[:80]
            break
    if not agency:
        m2 = _RE_AGENCY.search(head[:500])
        if m2:
            agency = m2.group(0)[:80].strip()
    if agency:
        meta.append({"field_name": "co_quan_ban_hanh", "value": agency, "confidence": 0.82})

    return meta

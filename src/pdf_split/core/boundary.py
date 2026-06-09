"""
Phát hiện ranh giới giữa các văn bản trong PDF:
- Dựa trên quốc hiệu, tên cơ quan, số văn bản
- Merge lại các segment quá ngắn không có dấu hiệu rõ ràng
"""
import re

from config.settings import RE_SO_VAN_BAN, MIN_SEG_PAGES


# ─── Pattern nhận diện đầu văn bản ───────────────────────────────────────────

_FIRST_LINE_PATTERN = re.compile(
    r'TÒA ÁN NHÂN DÂN|VIỆN KIỂM SÁT|CỘNG HÒA XÃ HỘI'
    r'|NHÂN DANH|QUYẾT ĐỊNH\s*$|ĐÌNH CHỈ|THÔNG BÁO'
)


def _is_doc_start(strict_header: str, first_line: str) -> tuple[bool, str]:
    """
    Kiểm tra trang có phải là trang đầu của một văn bản mới không.

    Trả về (is_start, signal_string) — signal_string mô tả các dấu hiệu tìm thấy.
    """
    h  = strict_header.upper()
    fl = first_line.upper()

    if not _FIRST_LINE_PATTERN.search(fl):
        return False, ""

    has_quoc_hieu  = "CỘNG HÒA XÃ HỘI" in h or "ĐỘC LẬP" in h or "HẠNH PHÚC" in h
    has_co_quan    = "TÒA ÁN NHÂN DÂN" in h or "VIỆN KIỂM SÁT" in h
    has_dinh_danh  = (
        bool(RE_SO_VAN_BAN.search(strict_header))
        or "BẢN ÁN SỐ" in h
        or "NHÂN DANH" in h
        or bool(re.search(r'QUYẾT ĐỊNH\s*$', h, re.M))
        or "ĐÌNH CHỈ VỤ ÁN" in h
        or "THÔNG BÁO THỤ LÝ" in h
    )

    threshold = 1 if (has_quoc_hieu and "CỘNG HÒA" in h) else 2
    if sum([has_quoc_hieu, has_co_quan, has_dinh_danh]) >= threshold:
        signals = "+".join(filter(None, [
            "quoc_hieu"  * has_quoc_hieu,
            "co_quan"    * has_co_quan,
            "dinh_danh"  * has_dinh_danh,
        ]))
        return True, signals

    return False, ""


# ─── Phát hiện ranh giới từng trang ──────────────────────────────────────────

def detect_boundary(feat: dict, prev_feat: dict | None) -> tuple[bool, float]:
    """
    Quyết định trang `feat` có bắt đầu một văn bản mới không.

    Trả về (is_boundary, confidence).
    """
    if prev_feat is None:
        return True, 1.0    # trang đầu tiên luôn là boundary

    if feat["is_blank"]:
        return False, 0.0   # trang trắng không tính là boundary

    if prev_feat["is_blank"]:
        return True, 0.95   # trang sau trang trắng → bắt đầu mới

    is_start, signal = _is_doc_start(*feat["strict_header"])
    if is_start:
        strong = "quoc_hieu" in signal and "co_quan" in signal
        return True, 1.0 if strong else 0.9

    # Số văn bản khác nhau hoàn toàn → ranh giới yếu
    prev_ids = set(prev_feat.get("so_vb_header", []))
    curr_ids = set(feat.get("so_vb_header", []))
    if curr_ids and prev_ids and not curr_ids & prev_ids:
        return True, 0.85

    return False, 0.0


# ─── Tạo danh sách segment ────────────────────────────────────────────────────

def get_segments(features: list[dict]) -> list[list[dict]]:
    """
    Chia danh sách trang thành các segment (mỗi segment = 1 văn bản).

    Quy tắc:
    - Trang trắng → kết thúc segment hiện tại, không thuộc segment nào.
    - Trang có boundary → bắt đầu segment mới.
    - Segment ≤ MIN_SEG_PAGES trang + không có dấu hiệu rõ → merge vào segment trước.
    """
    boundaries: set[int] = set()
    blank_pages: set[int] = set()

    for i, feat in enumerate(features):
        is_bd, _ = detect_boundary(feat, features[i - 1] if i > 0 else None)
        if is_bd:
            boundaries.add(feat["page_num"])
        if feat["is_blank"]:
            blank_pages.add(feat["page_num"])

    # Tạo segment thô
    raw_segments: list[list[dict]] = []
    current: list[dict] = []

    for feat in features:
        pn = feat["page_num"]
        if feat["is_blank"] or pn in blank_pages:
            if current:
                raw_segments.append(current)
                current = []
            continue
        if pn in boundaries and current:
            raw_segments.append(current)
            current = []
        current.append(feat)

    if current:
        raw_segments.append(current)

    # Merge segment ngắn không có dấu hiệu rõ vào segment trước
    if not raw_segments:
        return []

    merged = [raw_segments[0]]
    for seg in raw_segments[1:]:
        first_feat   = seg[0]
        header_upper = first_feat.get("strict_header", ("", ""))[0].upper()
        has_clear_start = (
            first_feat.get("has_quoc_hieu")
            or first_feat.get("has_toa_an")
            or bool(first_feat.get("so_vb_header"))
            or "NHÂN DANH" in header_upper
            or "BẢN ÁN SỐ" in header_upper
        )
        if len(seg) <= MIN_SEG_PAGES and not has_clear_start:
            merged[-1] += seg
        else:
            merged.append(seg)

    return merged

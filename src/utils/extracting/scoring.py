"""
scoring.py — Chấm điểm độ tin cậy (confidence) cho từng field trích xuất.

Tách ra từ notebook để:
  - Dễ unit-test độc lập với LLM.
  - Dễ tuning rule per-schema mà không đụng vào extractor.
  - Dễ mở rộng thêm rule mới (regex, cross-validate, business logic).

Export chính:
  compute_field_confidence(value, field_name) → float
  score_extraction(extracted_dict, schema_class, threshold) → (metadata_list, overall, fill_rate)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
# Thêm vào phần PATTERN LIBRARY (thay thế các _RE_ cũ):

_RE_SO_HIEU  = re.compile(
    r"^\d{1,4}"
    r"[\s/–\-]+"
    r"\d{4}"
    r"[\s/–\-]+"
    r"[A-ZĐÂÊÔĂƯĐ]{2,}"
)
_RE_DATE_VN   = re.compile(r"\d{1,2}\s*[/\-\.]\s*\d{1,2}\s*[/\-\.]\s*\d{4}")
_RE_DATE_TXT  = re.compile(r"\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", re.IGNORECASE)
_RE_DATE_DASH = re.compile(r"\d{1,2}\s*-\s*\d{1,2}\s*-\s*\d{4}")
_RE_TIME_VN   = re.compile(r"\d{1,2}\s*giờ.*ngày\s*\d{1,2}", re.IGNORECASE)
_RE_CCCD      = re.compile(r"^\d{9}$|^\d{12}$")
_RE_DIGIT     = re.compile(r"\d")
_RE_ANON      = re.compile(r"[A-ZĐĂÂÊÔƯĐ]\d+", re.UNICODE)


def compute_field_confidence(value: Any, field_name: str) -> float:
    if value is None or str(value).strip() == "":
        return 0.0
    v = str(value).strip()
    f = field_name.lower()

    # ── so_hieu / so_hop_dong ─────────────────────────────────────────────────
    if any(k in f for k in ("so_hieu", "so_ban_an", "thu_ly", "so_hop_dong")):
        if _RE_SO_HIEU.match(v):
            return 0.95
        if _RE_DIGIT.search(v) and re.search(r"[A-ZĐĂÂÊÔƯĐ]{2,}", v):
            return 0.80
        return 0.72 if _RE_DIGIT.search(v) else 0.45

    # ── ngày / thời gian ──────────────────────────────────────────────────────
    if any(k in f for k in ("ngay", "ngày", "date", "thoi_gian", "thời_gian")):
        if _RE_TIME_VN.search(v):
            return 0.90
        if _RE_DATE_TXT.search(v) or _RE_DATE_VN.search(v) or _RE_DATE_DASH.search(v):
            return 0.92
        if _RE_DIGIT.search(v):
            return 0.55
        return 0.30

    # ── năm sinh ──────────────────────────────────────────────────────────────
    if any(k in f for k in ("nam_sinh", "year")):
        try:
            return 0.95 if 1900 <= int(v) <= 2030 else 0.50
        except ValueError:
            return 0.30

    # ── CCCD / CMND ───────────────────────────────────────────────────────────
    if any(k in f for k in ("cccd", "cmnd")):
        return 0.95 if _RE_CCCD.match(v.replace(" ", "")) else (
            0.65 if _RE_DIGIT.search(v) else 0.35
        )

    # ── mã số BHXH / thẻ BHYT ────────────────────────────────────────────────
    if any(k in f for k in ("ma_so_bhxh", "the_bhyt", "ma_so")):
        clean = v.replace(" ", "")
        if re.match(r"^\d{10}$", clean):
            return 0.95
        if re.match(r"^[A-Z]{2}\d{13}$", clean):
            return 0.95
        return 0.65 if _RE_DIGIT.search(v) else 0.35

    # ── tên người ─────────────────────────────────────────────────────────────
    _NAME_KWS = (
        "ho_ten", "tham_phan", "thu_ky", "kiem_sat", "ten_bi",
        "nguyen_don", "bi_don", "nguoi_", "chu_tri", "nguoi_viet",
        "nguoi_duoc", "nguoi_ky",
    )
    if any(k in f for k in _NAME_KWS):
        if ";" in v:
            parts = [p.strip() for p in v.split(";") if p.strip()]
            ok = sum(1 for p in parts if len(p.split()) >= 2)
            return round(min(0.75 + 0.06 * ok, 0.95), 2)
        if len(v.split()) >= 2 and not _RE_DIGIT.search(v):
            return 0.90
        if _RE_ANON.search(v):
            return 0.82
        if len(v) >= 4:
            return 0.75
        return 0.60

    # ── loại văn bản / tên cơ quan ───────────────────────────────────────────
    if any(k in f for k in (
        "loai_van_ban", "loai_giay_to", "ten_bien_ban", "ten_toa_an",
        "co_quan", "toa_an", "toi_danh",
    )):
        words = len(v.split())
        if words >= 4:
            return 0.92
        if words >= 2:
            return 0.85
        return 0.75

    # ── nội dung / tóm tắt (default) ─────────────────────────────────────────
    words = len(v.split())
    return 0.85 if words >= 3 else (0.78 if words >= 2 else 0.65)


# ── Normalize helpers ─────────────────────────────────────────────────────────
_RE_DATE_NORMALIZE = [
    (re.compile(r"(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{4})"), r"\1/\2/\3"),
    (re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.I), r"\1/\2/\3"),
]
_RE_SOHIEU_NORMALIZE = re.compile(r"\s*[–—]\s*")

def _normalize_value(value: Any, fname: str) -> Any:
    if value is None or not isinstance(value, str):
        return value
    v, f = value.strip(), fname.lower()
    if any(k in f for k in ("ngay", "ngày", "date", "thoi_gian", "ngay_ky")):
        for pat, repl in _RE_DATE_NORMALIZE:
            v = pat.sub(repl, v)
    if any(k in f for k in ("so_hieu", "so_hop_dong", "so_ban_an")):
        v = _RE_SOHIEU_NORMALIZE.sub("-", v)
    return v


def score_extraction(
    extracted: Dict[str, Any],
    schema_class: Type[BaseModel],
    low_conf_threshold: float = 0.65,
) -> Tuple[List[dict], float, float]:
    """
    Phiên bản cập nhật: thêm _normalize_value trước khi tính confidence.
    """
    metadata_list: List[dict] = []
    confs: List[float] = []

    props = schema_class.model_json_schema().get("properties", {})

    for fname in props:
        raw_val = extracted.get(fname)
        val     = _normalize_value(raw_val, fname)   # normalize trước
        conf    = compute_field_confidence(val, fname)
        is_null = val is None or str(val).strip() == ""

        entry: dict = {
            "field_name": fname,
            "value":      val,
            "confidence": round(conf, 3) if not is_null else None,
            "flagged":    False if is_null else (conf < low_conf_threshold),
            "missing":    is_null,
        }
        metadata_list.append(entry)
        if not is_null:
            confs.append(conf)

    total     = len(metadata_list)
    filled    = sum(1 for m in metadata_list if not m["missing"])
    fill_rate = round(filled / total, 3) if total else 0.0
    overall   = round(sum(confs) / len(confs), 3) if confs else 0.0

    return metadata_list, overall, fill_rate
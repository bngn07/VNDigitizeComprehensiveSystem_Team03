from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

RE_SO_HIEU = re.compile(
    r"^\d{1,4}[\s/–\-]+\d{4}[\s/–\-]+[A-ZĐÂÊÔĂƯĐ]{2,}"
)
RE_DATE_VN = re.compile(r"\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{4}")
RE_DATE_TEXT = re.compile(r"\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", re.IGNORECASE)
RE_TIME_VN = re.compile(r"\d{1,2}\s*giờ.*ngày\s*\d{1,2}", re.IGNORECASE)
RE_CCCD = re.compile(r"^\d{9}$|^\d{12}$")
RE_DIGIT = re.compile(r"\d")
RE_ANON = re.compile(r"[A-ZĐĂÂÊÔƯĐ]\d+", re.UNICODE)


def compute_field_confidence(value: Any, field_name: str) -> float:
    if value is None or str(value).strip() == "":
        return 0.0

    value_str = str(value).strip()
    field = field_name.lower()

    if any(key in field for key in ("so_hieu", "so_ban_an", "thu_ly", "so_hop_dong")):
        if RE_SO_HIEU.match(value_str):
            return 0.95
        if RE_DIGIT.search(value_str) and re.search(r"[A-ZĐĂÂÊÔƯĐ]{2,}", value_str):
            return 0.80
        return 0.72 if RE_DIGIT.search(value_str) else 0.45

    if any(key in field for key in ("ngay", "date", "thoi_gian")):
        if RE_TIME_VN.search(value_str):
            return 0.90
        if RE_DATE_TEXT.search(value_str) or RE_DATE_VN.search(value_str):
            return 0.92
        if RE_DIGIT.search(value_str):
            return 0.55
        return 0.30

    if any(key in field for key in ("cccd", "cmnd")):
        compact = value_str.replace(" ", "")
        return 0.95 if RE_CCCD.match(compact) else 0.65 if RE_DIGIT.search(value_str) else 0.35

    name_keys = (
        "ho_ten",
        "tham_phan",
        "thu_ky",
        "kiem_sat",
        "nguyen_don",
        "bi_don",
        "nguoi_",
        "chu_tri",
        "nguoi_viet",
        "nguoi_duoc",
        "nguoi_ky",
    )
    if any(key in field for key in name_keys):
        if ";" in value_str:
            parts = [part.strip() for part in value_str.split(";") if part.strip()]
            valid = sum(1 for part in parts if len(part.split()) >= 2)
            return round(min(0.75 + 0.06 * valid, 0.95), 2)
        if len(value_str.split()) >= 2 and not RE_DIGIT.search(value_str):
            return 0.90
        if RE_ANON.search(value_str):
            return 0.82
        return 0.75 if len(value_str) >= 4 else 0.60

    words = len(value_str.split())
    return 0.85 if words >= 3 else 0.78 if words >= 2 else 0.65


def score_extraction(
    extracted: dict[str, Any],
    schema_class: type[BaseModel],
    low_conf_threshold: float = 0.65,
) -> tuple[list[dict[str, Any]], float, float]:
    metadata_list: list[dict[str, Any]] = []
    confidences: list[float] = []

    properties = schema_class.model_json_schema().get("properties", {})
    for field_name in properties:
        value = extracted.get(field_name)
        confidence = compute_field_confidence(value, field_name)
        missing = value is None or str(value).strip() == ""

        metadata_list.append(
            {
                "field_name": field_name,
                "value": value,
                "confidence": None if missing else round(confidence, 3),
                "flagged": False if missing else confidence < low_conf_threshold,
                "missing": missing,
            }
        )

        if not missing:
            confidences.append(confidence)

    total = len(metadata_list)
    filled = sum(1 for item in metadata_list if not item["missing"])
    fill_rate = round(filled / total, 3) if total else 0.0
    overall = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    return metadata_list, overall, fill_rate

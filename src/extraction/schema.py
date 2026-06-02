from __future__ import annotations

import re
from typing import Type

from pydantic import BaseModel, Field


class VanBanToaAnSchema(BaseModel):
    loai_van_ban: str | None = Field(default=None)
    so_hieu: str | None = Field(default=None)
    ngay_ban_hanh: str | None = Field(default=None)
    ten_toa_an: str | None = Field(default=None)


class DonTuCamKetSchema(BaseModel):
    loai_giay_to: str | None = Field(default=None)
    nguoi_viet_don_cam_ket: str | None = Field(default=None)
    noi_nhan_co_quan_giai_quyet: str | None = Field(default=None)
    ngay_thang_nam: str | None = Field(default=None)
    noi_dung_tom_tat: str | None = Field(default=None)


SCHEMA_REGISTRY: dict[str, Type[BaseModel]] = {
    "VAN_BAN_TOA_AN": VanBanToaAnSchema,
    "DON_TU_CAM_KET": DonTuCamKetSchema,
}

SCHEMA_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "VAN_BAN_TOA_AN",
        ["tòa án nhân dân", "bản án số", "thẩm phán", "xét xử"],
    ),
    (
        "DON_TU_CAM_KET",
        ["đơn tố cáo", "bản cam kết", "giấy xác nhận", "đơn khiếu nại"],
    ),
]


def auto_detect_schema(text: str) -> str:
    lowered = text.lower()

    for schema_name, keywords in SCHEMA_KEYWORDS:
        hits = sum(1 for keyword in keywords if keyword in lowered)
        if hits >= 2:
            return schema_name

    if re.search(r"bản\s+án|xét\s+xử|tuyên\s+phạt", lowered, re.IGNORECASE):
        return "VAN_BAN_TOA_AN"

    return "DON_TU_CAM_KET"

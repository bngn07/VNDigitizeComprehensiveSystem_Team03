from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class CourtRecord(BaseModel):
    so_ban_an:  Optional[str] = Field(
        default=None,
        description="Số bản án (case/judgment number), e.g. '746/2017/HS-PT'",
    )
    ten_bi_cao: Optional[str] = Field(
        default=None,
        description="Tên bị cáo (defendant's name), e.g. 'Đỗ Văn N'",
    )
    toi_danh:   Optional[str] = Field(
        default=None,
        description="Tội danh (charge / offence), e.g. 'Tội trộm cắp tài sản'",
    )
    nam_sinh:   Optional[int] = Field(
        default=None,
        description="Năm sinh (year of birth), e.g. 1948",
    )

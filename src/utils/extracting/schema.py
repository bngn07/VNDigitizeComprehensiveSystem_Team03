from __future__ import annotations
from pydantic import BaseModel, Field

class CourtRecord(BaseModel):
    so_ban_an:  str | None = Field(
        default=None,
        description="Số bản án (case/judgment number), e.g. '746/2017/HS-PT'",
    )
    ten_bi_cao: str | None = Field(
        default=None,
        description="Tên bị cáo (defendant's name), e.g. 'Đỗ Văn N'",
    )
    toi_danh:   str | None = Field(
        default=None,
        description="Tội danh (charge / offence), e.g. 'Tội trộm cắp tài sản'",
    )
    nam_sinh:   str | None = Field(
        default=None,
        description="",
    )

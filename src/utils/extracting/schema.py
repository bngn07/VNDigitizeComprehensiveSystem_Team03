from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Dict, Type, List, Tuple

class VanBanToaAnSchema(BaseModel):
    loai_van_ban:              Optional[str] = Field(default=None, description="VD: Bản án sơ thẩm, Quyết định công nhận thuận tình ly hôn")
    so_hieu:                   Optional[str] = Field(default=None, description="VD: 03/2022/DSST, 89/2026/QĐST-HNGĐ")
    ngay_ban_hanh:             Optional[str] = Field(default=None, description="VD: 23/11/2022")
    ten_toa_an:                Optional[str] = Field(default=None, description="VD: Tòa án nhân dân tỉnh BN")
    vu_viec_tom_tat:           Optional[str] = Field(default=None, description="Tóm tắt nội dung vụ việc")
    tham_phan_chu_toa:         Optional[str] = Field(default=None, description="Tên Thẩm phán hoặc Chủ tọa")
    nguyen_don_nguoi_khoi_kien:Optional[str] = Field(default=None, description="Tên Nguyên đơn / Người khởi kiện / Người yêu cầu")
    bi_don_nguoi_bi_kien:      Optional[str] = Field(default=None, description="Tên Bị đơn / Người bị kiện / Bị cáo")

class VanBanHanhChinhSchema(BaseModel):
    loai_van_ban:    Optional[str] = Field(default=None, description="VD: Quyết định, Thông báo, Kết luận thanh tra")
    so_hieu:         Optional[str] = Field(default=None, description="VD: 1059/QĐ-UBND")
    ngay_ban_hanh:   Optional[str] = Field(default=None, description="VD: 29 tháng 3 năm 2021")
    co_quan_ban_hanh:Optional[str] = Field(default=None, description="VD: UBND thành phố Đà Lạt")
    noi_dung_tom_tat:Optional[str] = Field(default=None, description="Về việc gì")
    nguoi_ky:        Optional[str] = Field(default=None, description="Tên người ký văn bản")

class BienBanSchema(BaseModel):
    ten_bien_ban:            Optional[str] = Field(default=None, description="VD: Biên bản ghi lời khai, vi phạm hành chính")
    thoi_gian_lap:           Optional[str] = Field(default=None, description="Thời gian lập biên bản")
    dia_diem_lap:            Optional[str] = Field(default=None, description="Nơi lập biên bản")
    nguoi_chu_tri_lap:       Optional[str] = Field(default=None, description="Tên cán bộ / người lập")
    nguoi_tham_gia_doi_tuong:Optional[str] = Field(default=None, description="Tên người được lấy lời khai / bị lập biên bản")
    noi_dung_chinh:          Optional[str] = Field(default=None, description="Tóm tắt nội dung biên bản")

class VanBanDieuTraSchema(BaseModel):
    loai_van_ban:       Optional[str] = Field(default=None, description="VD: Bản kết luận điều tra, Lệnh cấm đi khỏi nơi cư trú")
    so_hieu:            Optional[str] = Field(default=None, description="VD: 65/KLĐT-PC03")
    ngay_ban_hanh:      Optional[str] = Field(default=None, description="VD: 03 tháng 11 năm 2021")
    co_quan_ban_hanh:   Optional[str] = Field(default=None, description="VD: Cơ quan Cảnh sát điều tra Công an tỉnh Quảng Bình")
    ten_bi_can_doi_tuong:Optional[str] = Field(default=None, description="Họ tên bị can / đối tượng bị điều tra")
    toi_danh_vu_an:     Optional[str] = Field(default=None, description="Tội danh hoặc tên vụ án")

class DonTuCamKetSchema(BaseModel):
    loai_giay_to:              Optional[str] = Field(default=None, description="VD: Đơn tố cáo, Bản cam kết, Giấy xác nhận")
    nguoi_viet_don_cam_ket:    Optional[str] = Field(default=None, description="Tên người đứng đơn")
    noi_nhan_co_quan_giai_quyet:Optional[str]= Field(default=None, description="Kính gửi ai / cơ quan nào")
    ngay_thang_nam:            Optional[str] = Field(default=None, description="Ngày tháng ghi trên đơn")
    noi_dung_tom_tat:          Optional[str] = Field(default=None, description="Tóm tắt nội dung")

SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "VAN_BAN_TOA_AN":       VanBanToaAnSchema,
    "VAN_BAN_DIEU_TRA":     VanBanDieuTraSchema,
    "QUYET_DINH_HANH_CHINH":VanBanHanhChinhSchema,
    "BIEN_BAN":             BienBanSchema,
    "DON_TU_CAM_KET":       DonTuCamKetSchema,
}

SCHEMA_KEYWORDS: List[tuple] = [
    ("VAN_BAN_TOA_AN",        ["tòa án nhân dân","bản án số","quyết định đình chỉ xét xử","tòa phúc thẩm","tòa án","thẩm phán","xét xử"]),
    ("VAN_BAN_DIEU_TRA",      ["kết luận điều tra","cơ quan cảnh sát điều tra","viện kiểm sát","khởi tố bị can","lệnh cấm đi khỏi","truy tố","tội phạm"]),
    ("BIEN_BAN",              ["biên bản vi phạm","biên bản ghi lời khai","biên bản đối chất","biên bản thỏa thuận","biên bản giao nhận"]),
    ("QUYET_DINH_HANH_CHINH", ["quyết định","ủy ban nhân dân","ubnd","thanh tra tỉnh","thu hồi","sở tài nguyên","đình chỉ công tác"]),
    ("DON_TU_CAM_KET",        ["đơn tố cáo","bản cam kết","giấy cam kết","thư cảm ơn","giấy xác nhận","đơn khiếu nại","tôi tên là"]),
]

def auto_detect_schema(text: str) -> Tuple[str, Type[BaseModel]]:
    t = text.lower()
    for name, kws in SCHEMA_KEYWORDS:
        if any(k in t for k in kws):
            return name, SCHEMA_REGISTRY[name]
    return "DON_TU_CAM_KET", SCHEMA_REGISTRY["DON_TU_CAM_KET"]
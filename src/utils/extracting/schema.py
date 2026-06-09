from pydantic import BaseModel, Field
from typing import Optional, Dict, Type, List
import re


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class VanBanToaAnSchema(BaseModel):
    loai_van_ban:               Optional[str] = Field(default=None, description="VD: Bản án sơ thẩm, Quyết định công nhận thuận tình ly hôn")
    so_hieu:                    Optional[str] = Field(default=None, description="VD: 03/2022/DSST, 89/2026/QĐST-HNGĐ")
    ngay_ban_hanh:              Optional[str] = Field(default=None, description="VD: 23/11/2022")
    ten_toa_an:                 Optional[str] = Field(default=None, description="VD: Tòa án nhân dân tỉnh BN")
    vu_viec_tom_tat:            Optional[str] = Field(default=None, description="Tóm tắt nội dung vụ việc")
    tham_phan_chu_toa:          Optional[str] = Field(default=None, description="Tên Thẩm phán hoặc Chủ tọa")
    nguyen_don_nguoi_khoi_kien: Optional[str] = Field(default=None, description="Tên Nguyên đơn / Người khởi kiện / Người yêu cầu")
    bi_don_nguoi_bi_kien:       Optional[str] = Field(default=None, description="Tên Bị đơn / Người bị kiện / Bị cáo")

class VanBanHanhChinhSchema(BaseModel):
    loai_van_ban:     Optional[str] = Field(default=None, description="VD: Quyết định, Thông báo, Kết luận thanh tra")
    so_hieu:          Optional[str] = Field(default=None, description="VD: 1059/QĐ-UBND")
    ngay_ban_hanh:    Optional[str] = Field(default=None, description="VD: 29 tháng 3 năm 2021")
    co_quan_ban_hanh: Optional[str] = Field(default=None, description="VD: UBND thành phố Đà Lạt")
    noi_dung_tom_tat: Optional[str] = Field(default=None, description="Về việc gì")
    nguoi_ky:         Optional[str] = Field(default=None, description="Tên người ký văn bản")

class BienBanSchema(BaseModel):
    ten_bien_ban:             Optional[str] = Field(default=None, description="VD: Biên bản ghi lời khai, vi phạm hành chính")
    thoi_gian_lap:            Optional[str] = Field(default=None, description="Thời gian lập biên bản")
    dia_diem_lap:             Optional[str] = Field(default=None, description="Nơi lập biên bản")
    nguoi_chu_tri_lap:        Optional[str] = Field(default=None, description="Tên cán bộ / người lập")
    nguoi_tham_gia_doi_tuong: Optional[str] = Field(default=None, description="Tên người được lấy lời khai / bị lập biên bản")
    noi_dung_chinh:           Optional[str] = Field(default=None, description="Tóm tắt nội dung biên bản")

class VanBanDieuTraSchema(BaseModel):
    loai_van_ban:          Optional[str] = Field(default=None, description="VD: Bản kết luận điều tra, Lệnh cấm đi khỏi nơi cư trú")
    so_hieu:               Optional[str] = Field(default=None, description="VD: 65/KLĐT-PC03")
    ngay_ban_hanh:         Optional[str] = Field(default=None, description="VD: 03 tháng 11 năm 2021")
    co_quan_ban_hanh:      Optional[str] = Field(default=None, description="VD: Cơ quan Cảnh sát điều tra Công an tỉnh Quảng Bình")
    ten_bi_can_doi_tuong:  Optional[str] = Field(default=None, description="Họ tên bị can / đối tượng bị điều tra")
    toi_danh_vu_an:        Optional[str] = Field(default=None, description="Tội danh hoặc tên vụ án")

class DonTuCamKetSchema(BaseModel):
    loai_giay_to:               Optional[str] = Field(default=None, description="VD: Đơn tố cáo, Bản cam kết, Giấy xác nhận")
    nguoi_viet_don_cam_ket:     Optional[str] = Field(default=None, description="Tên người đứng đơn")
    noi_nhan_co_quan_giai_quyet:Optional[str] = Field(default=None, description="Kính gửi ai / cơ quan nào")
    ngay_thang_nam:             Optional[str] = Field(default=None, description="Ngày tháng ghi trên đơn")
    noi_dung_tom_tat:           Optional[str] = Field(default=None, description="Tóm tắt nội dung")

class VanBanBaoHiemSchema(BaseModel):
    loai_van_ban:        Optional[str] = Field(default=None, description="VD: Quyết định hưởng trợ cấp thất nghiệp, Thông báo đóng BHXH, Thẻ bảo hiểm")
    so_hieu:             Optional[str] = Field(default=None, description="VD: 1452/QĐ-BHXH, 4523/TB-BHXH")
    ngay_ban_hanh:       Optional[str] = Field(default=None, description="VD: 20/05/2026")
    co_quan_ban_hanh:    Optional[str] = Field(default=None, description="VD: Bảo hiểm xã hội quận Cầu Giấy, BHXH TP. Hồ Chí Minh")
    nguoi_duoc_bao_hiem: Optional[str] = Field(default=None, description="Họ và tên người tham gia / người được hưởng bảo hiểm")
    ma_so_bhxh_the_bhyt: Optional[str] = Field(default=None, description="Mã số BHXH hoặc Số thẻ BHYT (nếu có)")
    noi_dung_tom_tat:    Optional[str] = Field(default=None, description="Tóm tắt nội dung bảo hiểm (VD: hưởng trợ cấp 3 tháng, thông báo nợ tiền đóng)")

class HopDongSchema(BaseModel):
    loai_hop_dong:     Optional[str] = Field(default=None, description="VD: Hợp đồng thế chấp, Hợp đồng mua bán, Hợp đồng vay")
    so_hop_dong:       Optional[str] = Field(default=None, description="VD: 4552/HĐTC-VPB")
    ngay_ky:           Optional[str] = Field(default=None, description="Ngày ký hợp đồng")
    ben_a:             Optional[str] = Field(default=None, description="Bên A / Bên cho vay / Bên bán / Bên nhận thế chấp")
    ben_b:             Optional[str] = Field(default=None, description="Bên B / Bên vay / Bên mua / Bên thế chấp")
    tai_san_doi_tuong: Optional[str] = Field(default=None, description="Tài sản thế chấp / đối tượng hợp đồng")
    noi_dung_tom_tat:  Optional[str] = Field(default=None, description="Tóm tắt nội dung hợp đồng")
    nguoi_chung_thuc:  Optional[str] = Field(default=None, description="Công chứng viên / người chứng thực")


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY & KEYWORDS
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "VAN_BAN_TOA_AN":        VanBanToaAnSchema,
    "VAN_BAN_DIEU_TRA":      VanBanDieuTraSchema,
    "QUYET_DINH_HANH_CHINH": VanBanHanhChinhSchema,
    "BIEN_BAN":              BienBanSchema,
    "DON_TU_CAM_KET":        DonTuCamKetSchema,
    "VAN_BAN_BAO_HIEM":      VanBanBaoHiemSchema,
    "HOP_DONG":              HopDongSchema,
}

SCHEMA_KEYWORDS: List[tuple] = [
    ("VAN_BAN_TOA_AN",        ["tòa án nhân dân","bản án số","quyết định đình chỉ xét xử","tòa phúc thẩm","tòa án","thẩm phán","xét xử"]),
    ("VAN_BAN_DIEU_TRA",      ["kết luận điều tra","cơ quan cảnh sát điều tra","viện kiểm sát","khởi tố bị can","lệnh cấm đi khỏi","truy tố","tội phạm"]),
    ("BIEN_BAN",              ["biên bản vi phạm","biên bản ghi lời khai","biên bản đối chất","biên bản thỏa thuận","biên bản giao nhận"]),
    ("QUYET_DINH_HANH_CHINH", ["quyết định","ủy ban nhân dân","ubnd","thanh tra tỉnh","thu hồi","sở tài nguyên","đình chỉ công tác"]),
    ("VAN_BAN_BAO_HIEM",      ["bảo hiểm xã hội","bhxh","bảo hiểm y tế","bhyt","trợ cấp thất nghiệp","sổ bảo hiểm"]),
    ("HOP_DONG",              ["hợp đồng","bên a","bên b","công chứng","thế chấp","bên vay","bên cho vay","văn phòng công chứng"]),
    ("DON_TU_CAM_KET",        ["đơn tố cáo","bản cam kết","giấy cam kết","thư cảm ơn","giấy xác nhận","đơn khiếu nại","tôi tên là"]),
]

_DOCID_PREFIX_SCHEMA: List[tuple] = [
    ("toa_an",     "VAN_BAN_TOA_AN"),
    ("bien_ban",   "BIEN_BAN"),
    ("dieu_tra",   "VAN_BAN_DIEU_TRA"),
    ("hanh_chinh", "QUYET_DINH_HANH_CHINH"),
    ("bao_hiem",   "VAN_BAN_BAO_HIEM"),
    ("don_tu",     "DON_TU_CAM_KET"),
    ("cam_ket",    "DON_TU_CAM_KET"),
    ("hop_dong",   "HOP_DONG"),
    ("dan_su",     "HOP_DONG"),
]


# ══════════════════════════════════════════════════════════════════════════════
# AUTO DETECT SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

def auto_detect_schema(text: str) -> str:

    t = text.lower()
    header = t[:300]

    _HEADER_RULES = [
        ("VAN_BAN_BAO_HIEM",      [r"(bhxh|bhyt|bảo\s*hiểm\s*(xã\s*hội|y\s*tế)|thẻ\s*bảo\s*hiểm|trợ\s*cấp\s*thất\s*nghiệp)"]),
        ("HOP_DONG",              [r"(hợp\s*đồng\s*(thế\s*chấp|mua\s*bán|vay|dịch\s*vụ|lao\s*động)|văn\s*phòng\s*công\s*chứng)"]),
        ("BIEN_BAN",              [r"biên\s*bản\s*(vi\s*phạm|ghi\s*lời|đối\s*chất|khám\s*nghiệm|thỏa\s*thuận|giao\s*nhận)"]),
        ("VAN_BAN_DIEU_TRA",      [r"(bản\s*kết\s*luận\s*điều\s*tra|cáo\s*trạng|lệnh\s*(bắt|tạm\s*giam|khám\s*xét))"]),
        ("VAN_BAN_TOA_AN",        [r"(bản\s*án\s*số|nhân\s*danh\s*nước\s*cộng\s*hòa|tòa\s*(phúc\s*thẩm|án\s*nhân\s*dân))"]),
        ("QUYET_DINH_HANH_CHINH", [r"(quyết\s*định|kết\s*luận\s*thanh\s*tra|giấy\s*phép\s*xây\s*dựng)"]),
        ("DON_TU_CAM_KET",        [r"(đơn\s*(khiếu\s*nại|tố\s*cáo|yêu\s*cầu)|bản\s*cam\s*kết|tôi\s+tên\s+là)"]),
    ]
    for schema_name, patterns in _HEADER_RULES:
        for pat in patterns:
            if re.search(pat, header, re.IGNORECASE):
                return schema_name

    scores: Dict[str, int] = {}
    for name, kws in SCHEMA_KEYWORDS:
        hits = sum(1 for k in kws if k in t)
        if hits > 0:
            scores[name] = hits

    if scores:
        best_name  = max(scores, key=lambda n: scores[n])
        best_score = scores[best_name]
        if best_score >= 2:
            return best_name
        top = [n for n, s in scores.items() if s == best_score]
        if len(top) == 1:
            return best_name
        for name, _ in SCHEMA_KEYWORDS:
            if name in top:
                return name

    if re.search(r"bản\s+án|xét\s+xử|tuyên\s+phạt", t, re.IGNORECASE):
        return "VAN_BAN_TOA_AN"
    if re.search(r"điều\s+tra|khởi\s+tố|truy\s+tố", t, re.IGNORECASE):
        return "VAN_BAN_DIEU_TRA"
    if re.search(r"biên\s+bản", t, re.IGNORECASE):
        return "BIEN_BAN"
    if re.search(r"bhxh|bhyt|bảo\s+hiểm", t, re.IGNORECASE):
        return "VAN_BAN_BAO_HIEM"
    if re.search(r"\d{1,4}/\d{4}/[A-ZĐĂÂÊÔƯĐ]{2,}", t):
        return "QUYET_DINH_HANH_CHINH"

    return "DON_TU_CAM_KET"
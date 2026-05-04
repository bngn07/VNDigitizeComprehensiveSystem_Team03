"""
VN-Digitize Comprehensive Test Data Generator
================================================
Phủ đầy đủ tất cả yêu cầu kỹ thuật trong tài liệu AI Requirements:

  Module 1 - Image Pre-processing  : deskew, auto-crop, yellowing, noise, faded,
                                      blank page, upside-down, QR/barcode
  Module 2 - Core OCR Engine       : chữ in + chữ viết tay giả lập, ground-truth
                                      bounding boxes, confidence scoring
  Module 3 - Dynamic NER           : 4 khối nghiệp vụ (Hành chính, Tòa án,
                                      Bảo hiểm, Công an), metadata động
  Module 4 - Advanced NLP          : multi-document PDF bundles cho PDF Splitting,
                                      manifest JSON chuẩn API contract

Cấu trúc output:
  dataset_vn_digitize/
    images/          ← ảnh JPEG đã qua degradation
    ground_truth/    ← JSON chuẩn API contract (bounding_box, confidence...)
    pdf_bundles/     ← multi-page PDF + manifest cho Module 4

Usage:
    pip install pillow opencv-python-headless numpy qrcode[pil]
    python document_generator.py
"""

import os
import cv2
import csv
import json
import math
import random
import uuid
import urllib.request
from datetime import date, timedelta

import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS / FAKE DATA
# ─────────────────────────────────────────────────────────────────────────────

A4_W, A4_H = 1240, 1754          # pixels @150dpi ≈ A4

FAKE_NAMES = [
    "Nguyễn Văn An", "Trần Thị Bình", "Lê Minh Châu", "Phạm Quốc Dũng",
    "Hoàng Thị Em", "Vũ Văn Phong", "Đặng Minh Quân", "Bùi Thị Hoa",
    "Ngô Thanh Sơn", "Đinh Thị Lan", "Lý Văn Hùng", "Mai Thị Xuân",
]

FAKE_ORGS = [
    "Công ty CP Công nghệ ABC Việt Nam",
    "Tập đoàn Xây dựng XYZ",
    "HTX Nông nghiệp Hòa Bình",
    "Trường THPT Nguyễn Du",
    "Bệnh viện Đa khoa Trung Ương",
    "Công ty TNHH Tư vấn Pháp luật DEF",
]

DISTRICTS = ["Quận 1", "Quận 3", "Quận 7", "Quận Bình Thạnh",
             "Huyện Nhà Bè", "TP. Thủ Đức"]

STREETS = ["Lê Lợi", "Nguyễn Huệ", "Trần Hưng Đạo", "Điện Biên Phủ",
           "Hoàng Diệu", "Phan Văn Trị", "Cách Mạng Tháng 8"]

INSURANCE_ORGS = [
    "BẢO HIỂM XÃ HỘI VIỆT NAM",
    "CÔNG TY BẢO HIỂM BẢO VIỆT",
    "PRUDENTIAL VIỆT NAM ASSURANCE",
    "MANULIFE VIỆT NAM",
    "AIA VIỆT NAM",
]

TOI_DANH_LIST = [
    ("Lừa đảo chiếm đoạt tài sản",      "Khoản 2, Điều 174 BLHS 2015", "5–10 năm"),
    ("Cố ý gây thương tích",             "Khoản 3, Điều 134 BLHS 2015", "5–10 năm"),
    ("Trộm cắp tài sản",                 "Khoản 2, Điều 173 BLHS 2015", "3–7 năm"),
    ("Tham nhũng",                       "Điều 354 BLHS 2015",          "7–15 năm"),
    ("Vi phạm quy định quản lý đất đai", "Điều 229 BLHS 2015",          "2–7 năm"),
    ("Gây rối trật tự công cộng",        "Khoản 2, Điều 318 BLHS 2015", "2–7 năm"),
]

VI_PHAM_GTDB = [
    ("Vượt đèn đỏ",           "Khoản 5, Điều 6 NĐ 100/2019/NĐ-CP",       "4.000.000"),
    ("Không đội mũ bảo hiểm", "Khoản 2, Điều 8 NĐ 100/2019/NĐ-CP",       "300.000"),
    ("Dùng điện thoại khi lái xe", "Điểm i, K3, Điều 6 NĐ 100/2019/NĐ-CP","1.000.000"),
    ("Nồng độ cồn vượt mức",  "Điểm c, K8, Điều 5 NĐ 100/2019/NĐ-CP",    "35.000.000"),
    ("Chạy quá tốc độ >20km/h","Khoản 4, Điều 6 NĐ 100/2019/NĐ-CP",      "4.000.000"),
    ("Đi ngược chiều",        "Khoản 3, Điều 6 NĐ 100/2019/NĐ-CP",       "3.000.000"),
]

VEHICLE_BRANDS = ["Honda", "Yamaha", "Toyota", "Ford", "Vinfast", "Kia", "Hyundai"]
VEHICLE_TYPES  = ["Mô tô", "Xe máy điện", "Ô tô con", "Xe tải nhỏ", "Xe bán tải"]
LICENSE_CLASSES = ["A1", "A2", "B1", "B2", "C", "D", "E"]


# ─────────────────────────────────────────────────────────────────────────────
# FONT LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_fonts():
    """
    Tự động tải font Roboto (hỗ trợ đầy đủ Tiếng Việt) từ Google Fonts 
    nếu hệ thống chưa có, tránh lỗi mất chữ, thiếu dấu.
    """
    font_dir = "fonts"
    os.makedirs(font_dir, exist_ok=True)
    
    # Sử dụng link raw trực tiếp từ repo source của Roboto (luôn ổn định và tồn tại)
    bold_url = "https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Bold.ttf"
    reg_url = "https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf"
    
    bold_path = os.path.join(font_dir, "Roboto-Bold.ttf")
    reg_path = os.path.join(font_dir, "Roboto-Regular.ttf")
    
    def download_font(url, save_path):
        if not os.path.exists(save_path):
            print(f"Đang tải font {os.path.basename(save_path)} (Hỗ trợ Tiếng Việt)...")
            # Thêm User-Agent để tránh bị GitHub chặn request bằng lỗi 403/404
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
                out_file.write(response.read())

    # Tải font nếu chưa có trong máy
    try:
        download_font(bold_url, bold_path)
        download_font(reg_url, reg_path)
    except Exception as e:
        print(f"Cảnh báo: Không thể tự động tải font: {e}")

    # Ưu tiên load font Roboto vừa tải, nếu lỗi sẽ fallback về font hệ thống
    bold_candidates = [
        bold_path,
        "arialbd.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    
    regular_candidates = [
        reg_path,
        "arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]

    def try_load(paths, size):
        for p in paths:
            try:
                return ImageFont.truetype(p, size)
            except (IOError, OSError):
                continue
        return ImageFont.load_default()

    return {
        "title":    try_load(bold_candidates, 38),
        "subtitle": try_load(bold_candidates, 28),
        "label":    try_load(bold_candidates, 22),
        "body":     try_load(regular_candidates, 22),
        "small":    try_load(regular_candidates, 18),
        "hand":     try_load(regular_candidates, 23),   # giả lập chữ viết tay
    }


FONTS = _load_fonts()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER DRAWING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _rand_name() -> str:
    return random.choice(FAKE_NAMES)

def _rand_date(year_start=2018, year_end=2024) -> str:
    start = date(year_start, 1, 1)
    delta = (date(year_end, 12, 31) - start).days
    d = start + timedelta(days=random.randint(0, delta))
    return f"ngày {d.day:02d} tháng {d.month:02d} năm {d.year}"

def _rand_doc_no(prefix: str, org: str) -> str:
    return f"{random.randint(10, 999)}/{prefix}-{org}"

def _rand_cccd() -> str:
    return str(random.randint(10**11, 10**12 - 1))

def _rand_phone() -> str:
    return f"0{random.randint(300_000_000, 999_999_999)}"

def _rand_address() -> str:
    return (f"Số {random.randint(1, 999)} đường {random.choice(STREETS)}, "
            f"{random.choice(DISTRICTS)}, TP. HCM")

def _rand_bien_so() -> str:
    letters = "ABCDEFGHKLMNPSTUVXY"
    return (f"{random.randint(10, 99)}-{random.choice(letters)}{random.choice(letters)}"
            f" {random.randint(10000, 99999)}")

def _add_qr(img_pil: Image.Image, data: str, xy=(950, 80)) -> tuple:
    """Dán QR Code vào ảnh và trả về (x1,y1,x2,y2) bounding box."""
    try:
        qr = qrcode.QRCode(box_size=4, border=2)
        qr.add_data(data)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        if hasattr(qr_img, "get_image"):
            qr_img = qr_img.get_image()
        elif not isinstance(qr_img, Image.Image):
            qr_img = qr_img._img
        img_pil.paste(qr_img, xy)
        w, h = qr_img.size
        return (xy[0], xy[1], xy[0] + w, xy[1] + h)
    except Exception:
        return None

def _add_random_qr(img_pil: Image.Image, data: str, prob: float = 0.65) -> tuple:
    """Thêm QR ngẫu nhiên (có/không) và ngẫu nhiên vị trí (góc dưới trái, góc trên phải, bất kỳ)."""
    # Không phải văn bản nào cũng có mã QR (xác suất 65% có)
    if random.random() > prob:
        return None
    
    positions = [
        (80, A4_H - 180),  # Cố định góc dưới bên trái
        (960, 70),         # Cố định góc trên bên phải
        (random.randint(80, A4_W - 200), random.randint(70, A4_H - 200)) # Ngẫu nhiên hoàn toàn
    ]
    # Lựa chọn ngẫu nhiên 1 trong 3 vị trí trên
    return _add_qr(img_pil, data, xy=random.choice(positions))

def _draw_seal(draw: ImageDraw.Draw, cx: int, cy: int, color=(160, 0, 0)):
    """Vẽ con dấu tròn giả lập."""
    draw.ellipse([cx-65, cy-65, cx+65, cy+65], outline=color, width=3)
    draw.ellipse([cx-55, cy-55, cx+55, cy+55], outline=color, width=1)
    draw.text((cx-42, cy-14), "ỦY BAN", font=FONTS["small"], fill=color)
    draw.text((cx-38, cy+2),  "NHÂN DÂN", font=FONTS["small"], fill=color)

def _draw_signature(draw: ImageDraw.Draw, x: int, y: int, name: str,
                    title: str = "", pen_color=(0, 0, 180)):
    """Vẽ khối chữ ký gồm chức danh + nét ký giả lập + tên."""
    if title:
        draw.text((x, y), title, font=FONTS["label"], fill=(0, 0, 0))
        y += 30
    # Nét ký: đường sin lượn sóng
    sig_y = y + 20
    pts = [(x + i * 4, sig_y + int(10 * math.sin(i * 0.5)) + random.randint(-3, 3))
           for i in range(25)]
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=pen_color, width=2)
    draw.text((x, y + 48), name, font=FONTS["body"], fill=(0, 0, 0))

def _draw_table(draw: ImageDraw.Draw, x: int, y: int,
                headers: list, rows: list, col_widths: list,
                row_h: int = 36) -> int:
    """Vẽ bảng đơn giản, trả về y sau bảng."""
    def draw_row(cols, row_y, font, fill_bg=None):
        cx = x
        for txt, w in zip(cols, col_widths):
            if fill_bg:
                draw.rectangle([cx, row_y, cx + w, row_y + row_h], fill=fill_bg)
            draw.rectangle([cx, row_y, cx + w, row_y + row_h],
                           outline=(0, 0, 0), width=1)
            draw.text((cx + 5, row_y + 8), str(txt), font=font, fill=(0, 0, 0))
            cx += w

    draw_row(headers, y, FONTS["label"], fill_bg=(220, 220, 220))
    y += row_h
    for row in rows:
        draw_row(row, y, FONTS["small"])
        y += row_h
    return y

def _multiline(draw: ImageDraw.Draw, text: str, x: int, y: int,
               font=None, fill=(0, 0, 0), line_h: int = 30) -> int:
    """Vẽ text nhiều dòng, trả về y sau cùng."""
    if font is None:
        font = FONTS["body"]
    for line in text.split("\n"):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y

def _section_title(draw: ImageDraw.Draw, text: str, x: int, y: int) -> int:
    """Vẽ tiêu đề section in đậm, có gạch chân."""
    draw.text((x, y), text, font=FONTS["label"], fill=(0, 0, 0))
    tw = draw.textlength(text, font=FONTS["label"])
    draw.line((x, y + 26, x + tw, y + 26), fill=(0, 0, 0), width=1)
    return y + 38

def _header_block(draw: ImageDraw.Draw,
                  left_org: str, left_sub: str, doc_no: str,
                  location_date: str) -> int:
    """Vẽ header chuẩn văn bản hành chính VN, trả về y tiếp theo."""
    # Left: cơ quan ban hành
    draw.text((80, 60),  left_org.upper(), font=FONTS["subtitle"], fill=(0, 0, 0))
    draw.text((80, 98),  left_sub.upper(), font=FONTS["label"],    fill=(0, 0, 0))
    draw.text((80, 132), f"Số: {doc_no}",  font=FONTS["body"],     fill=(0, 0, 0))
    # Right: quốc hiệu
    draw.text((640, 60),  "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", font=FONTS["body"], fill=(0, 0, 0))
    draw.text((730, 98),  "Độc lập - Tự do - Hạnh phúc",         font=FONTS["body"], fill=(0, 0, 0))
    draw.line((730, 128, 1160, 128), fill=(0, 0, 0), width=2)
    draw.text((800, 138), location_date,                          font=FONTS["small"], fill=(0, 0, 0))
    # Divider
    draw.line((80, 180, 1160, 180), fill=(0, 0, 0), width=1)
    return 210

def _doc_title_block(draw: ImageDraw.Draw, doc_type: str, about: str,
                     start_y: int) -> int:
    """Vẽ tên loại văn bản + trích yếu, căn giữa."""
    tw = draw.textlength(doc_type, font=FONTS["title"])
    draw.text(((A4_W - tw) // 2, start_y), doc_type, font=FONTS["title"], fill=(0, 0, 0))
    start_y += 52
    tw2 = draw.textlength(f"V/v: {about}", font=FONTS["body"])
    draw.text(((A4_W - tw2) // 2, start_y), f"V/v: {about}", font=FONTS["body"], fill=(0, 0, 0))
    draw.line(((A4_W - tw2) // 2, start_y + 30, (A4_W + tw2) // 2, start_y + 30),
              fill=(0, 0, 0), width=1)
    return start_y + 55


def _form_field(draw: ImageDraw.Draw, label: str, value: str,
                lx: int, ly: int, vx: int, line_end_x: int,
                handwritten: bool = False) -> tuple:
    """
    Vẽ một dòng biểu mẫu: nhãn in đậm + giá trị (chữ in hoặc chữ tay giả lập).
    Trả về bounding box của phần value: (x1, y1, x2, y2).
    """
    draw.text((lx, ly), label, font=FONTS["label"], fill=(0, 0, 0))
    font = FONTS["hand"] if handwritten else FONTS["body"]
    ink  = (0, 0, 180) if handwritten else (0, 0, 0)   # xanh = viết tay
    draw.text((vx, ly), value, font=font, fill=ink)
    draw.line((vx, ly + 28, line_end_x, ly + 28), fill=(180, 180, 180), width=1)
    # approx bounding box cho ground truth
    val_w = int(draw.textlength(value, font=font))
    return (vx, ly, vx + val_w, ly + 28)


# ─────────────────────────────────────────────────────────────────────────────
# GROUND-TRUTH RECORD  (chuẩn API contract Module 5)
# ─────────────────────────────────────────────────────────────────────────────

def _make_gt(doc_id: str, classification: str, sector: str,
             summary: str, fields: dict) -> dict:
    """
    Tạo ground-truth JSON chuẩn API contract.
    fields: { field_name: (value, (x1,y1,x2,y2)) }
    """
    metadata = []
    for fname, (fval, bbox) in fields.items():
        metadata.append({
            "field_name":   fname,
            "value":        fval,
            "confidence":   round(random.uniform(0.82, 0.99), 2),   # synthetic
            "bounding_box": {"x1": bbox[0], "y1": bbox[1],
                             "x2": bbox[2], "y2": bbox[3]},
        })
    return {
        "document_id":        doc_id,
        "sector":             sector,
        "classification":     classification,
        "summary":            summary,
        "confidence_overall": round(random.uniform(0.88, 0.98), 2),
        "metadata":           metadata,
        "sub_documents":      [],           # sẽ điền cho PDF bundles
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 – DOCUMENT GENERATORS (4 sectors)
# ─────────────────────────────────────────────────────────────────────────────

def gen_hanh_chinh() -> tuple:
    """Văn bản hành chính: Quyết định, Thông báo, Tờ trình, Biên bản, Công văn."""
    doc_types = ["QUYẾT ĐỊNH", "THÔNG BÁO", "TỜ TRÌNH", "BIÊN BẢN", "CÔNG VĂN", "CHỈ THỊ"]
    doc_type  = random.choice(doc_types)
    org_top   = random.choice(["UBND", "SỞ NỘI VỤ", "PHÒNG TƯ PHÁP", "BAN QUẢN LÝ"])
    org_sub   = f"{org_top} QUẬN/HUYỆN XYZ"
    doc_no    = _rand_doc_no(doc_type[:2], org_top.split()[0])
    doc_date  = _rand_date()
    about     = "Phê duyệt kế hoạch số hóa tài liệu lưu trữ giai đoạn 2024–2026"
    doc_id    = uuid.uuid4().hex[:10].upper()

    img  = Image.new("RGB", (A4_W, A4_H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    fields_gt = {}

    y = _header_block(draw, org_top, org_sub, doc_no,
                      f"TP. Hồ Chí Minh, {doc_date}")

    # doc_no bounding box (approx row 3 of header)
    fields_gt["so_van_ban"]       = (doc_no,    (80, 132, 80 + 300, 160))
    fields_gt["ngay_thang_nam"]   = (doc_date,  (800, 138, 1160, 166))
    fields_gt["co_quan_ban_hanh"] = (org_sub,   (80, 98, 80 + 400, 126))

    y = _doc_title_block(draw, doc_type, about, y)
    fields_gt["ten_loai_van_ban"] = (doc_type, (350, y - 107, 950, y - 55))
    fields_gt["trich_yeu"]        = (about,    (300, y - 55,  1000, y - 25))

    # Căn cứ
    y = _section_title(draw, "CĂN CỨ PHÁP LÝ:", 80, y)
    can_cu = [
        "- Luật Lưu trữ ngày 11 tháng 11 năm 2011;",
        "- Nghị định số 30/2020/NĐ-CP ngày 05/3/2020 của Chính phủ về công tác văn thư;",
        "- Quyết định số 458/QĐ-TTg ngày 03/4/2020 của Thủ tướng Chính phủ về phê duyệt",
        "  Đề án lưu trữ tài liệu điện tử của các cơ quan nhà nước giai đoạn 2020–2025;",
        f"- Đề nghị của Giám đốc {org_sub}.",
    ]
    for cc in can_cu:
        draw.text((90, y), cc, font=FONTS["body"], fill=(0, 0, 0))
        y += 30
    y += 15

    # Nội dung quyết định
    tw = draw.textlength(f"NAY {doc_type}:", FONTS["label"])
    draw.text(((A4_W - tw) // 2, y), f"NAY {doc_type}:", font=FONTS["label"], fill=(0, 0, 0))
    y += 45

    don_vi = random.choice(FAKE_ORGS)
    nguoi_thuc_hien = _rand_name()
    articles = [
        ("Điều 1.",
         f"Phê duyệt Kế hoạch số hóa tài liệu lưu trữ của {org_sub} giai đoạn 2024–2026,\n"
         f"với tổng kinh phí dự kiến {random.randint(1, 5)}.{random.randint(100,900)}.000.000 đồng."),
        ("Điều 2.",
         f"Giao {don_vi} tổ chức triển khai thực hiện đúng tiến độ theo kế hoạch\n"
         "được phê duyệt, đảm bảo chất lượng và tuân thủ các quy định hiện hành."),
        ("Điều 3.",
         f"Ông/Bà {nguoi_thuc_hien}, Trưởng phòng Hành chính, các phòng ban liên quan\n"
         "chịu trách nhiệm thi hành Quyết định này kể từ ngày ký ban hành."),
    ]
    for art_no, art_body in articles:
        draw.text((90, y), art_no, font=FONTS["label"], fill=(0, 0, 0))
        for i, line in enumerate(art_body.split("\n")):
            draw.text((220, y + i * 30), line, font=FONTS["body"], fill=(0, 0, 0))
        y += 30 * len(art_body.split("\n")) + 20

    y += 30
    # Nơi nhận (trái)
    draw.text((80, y), "Nơi nhận:", font=FONTS["label"], fill=(0, 0, 0))
    for nn in ["- Như Điều 3;", "- UBND TP. HCM (để b/c);", "- Lưu VT, KHTC."]:
        y += 28
        draw.text((80, y), nn, font=FONTS["small"], fill=(0, 0, 0))

    # Chữ ký (phải)
    sig_y = y - 80
    _draw_signature(draw, 820, sig_y, _rand_name(), title="TM. ỦY BAN NHÂN DÂN\nCHỦ TỊCH")
    _draw_seal(draw, 870, sig_y + 170)

    # QR code
    qr_bb = _add_random_qr(img, f"DOC:{doc_id}:{doc_no}", prob=0.65)

    gt = _make_gt(doc_id, doc_type, "hanh_chinh",
                  f"Phê duyệt kế hoạch số hóa tài liệu của {org_sub}", fields_gt)
    return img, gt


def gen_toa_an() -> tuple:
    """Văn bản Tòa án: Bản án, Quyết định, Đơn khởi kiện, Biên bản hòa giải."""
    doc_types = ["BẢN ÁN", "QUYẾT ĐỊNH ĐÌNH CHỈ", "ĐƠN KHỞI KIỆN",
                 "BIÊN BẢN HÒA GIẢI", "LỆNH BẮT TẠM GIAM"]
    doc_type  = random.choice(doc_types)
    so_vu     = f"{random.randint(10,999)}/{random.randint(2020,2024)}/HS-ST"
    doc_date  = _rand_date()
    doc_id    = uuid.uuid4().hex[:10].upper()

    bi_cao        = _rand_name()
    nguyen_don    = _rand_name()
    toi_info      = random.choice(TOI_DANH_LIST)
    toi_danh      = toi_info[0]
    dieu_luat     = toi_info[1]
    muc_phat      = toi_info[2]
    bi_cao_cccd   = _rand_cccd()
    bi_cao_ngaysinh = f"{random.randint(1,28)}/{random.randint(1,12)}/{random.randint(1970,2002)}"
    bi_cao_addr   = _rand_address()

    img  = Image.new("RGB", (A4_W, A4_H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    fields_gt = {}

    # Header – Tòa án
    draw.text((80, 60),  "TÒA ÁN NHÂN DÂN THÀNH PHỐ HỒ CHÍ MINH",
              font=FONTS["subtitle"], fill=(0, 0, 0))
    draw.text((640, 60), "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
              font=FONTS["body"], fill=(0, 0, 0))
    draw.text((730, 98), "Độc lập - Tự do - Hạnh phúc",
              font=FONTS["body"], fill=(0, 0, 0))
    draw.line((730, 128, 1160, 128), fill=(0, 0, 0), width=2)
    draw.line((80, 180, 1160, 180), fill=(0, 0, 0), width=1)

    y = 210
    tw = draw.textlength(doc_type, FONTS["title"])
    draw.text(((A4_W - tw) // 2, y), doc_type, font=FONTS["title"], fill=(0, 0, 0))
    y += 50

    no_lbl = f"Số: {so_vu}"
    draw.text((80, y), no_lbl, font=FONTS["label"], fill=(0, 0, 0))
    draw.text((700, y), f"TP. Hồ Chí Minh, {doc_date}", font=FONTS["small"], fill=(0, 0, 0))
    fields_gt["so_vu_an"]         = (so_vu,    (80,  y,  80+350,  y+28))
    fields_gt["ngay_xet_xu"]      = (doc_date, (700, y,  1160,    y+28))
    fields_gt["co_quan_ban_hanh"] = ("TAND TP. Hồ Chí Minh", (80, 60, 700, 88))
    fields_gt["ten_loai_van_ban"] = (doc_type, ((A4_W-tw)//2, 210, (A4_W+tw)//2, 258))
    y += 50

    # Thành phần xét xử
    y = _section_title(draw, "THÀNH PHẦN XÉT XỬ:", 80, y)
    thanh_phan = [
        ("Thẩm phán chủ tọa phiên tòa:", _rand_name()),
        ("Hội thẩm nhân dân:",           _rand_name()),
        ("Kiểm sát viên (VKSND):",       _rand_name()),
        ("Thư ký phiên tòa:",            _rand_name()),
    ]
    for lbl, val in thanh_phan:
        draw.text((90,  y), lbl, font=FONTS["label"], fill=(0, 0, 0))
        draw.text((430, y), val, font=FONTS["body"],  fill=(0, 0, 0))
        y += 32
    y += 10

    # Bị cáo
    y = _section_title(draw, "THÔNG TIN BỊ CÁO:", 80, y)
    bc_fields = [
        ("Họ và tên:",          bi_cao,          True),
        ("Ngày sinh:",          bi_cao_ngaysinh, True),
        ("CCCD/CMND số:",       bi_cao_cccd,     True),
        ("Địa chỉ thường trú:", bi_cao_addr,     True),
        ("Nghề nghiệp:",        random.choice(["Buôn bán","Công nhân","Lái xe","Tự do"]), True),
    ]
    for lbl, val, hw in bc_fields:
        bb = _form_field(draw, lbl, val, 90, y, 380, 1150, handwritten=hw)
        y += 38
    fields_gt["ten_bi_cao"] = (bi_cao,      (380, y - 38*5, 380+300, y - 38*4))
    fields_gt["cmnd_cccd"]  = (bi_cao_cccd, (380, y - 38*3, 380+300, y - 38*2))
    y += 10

    # Tội danh
    y = _section_title(draw, "TỘI DANH BỊ TRUY TỐ:", 80, y)
    draw.text((90, y),       f"Tội:      {toi_danh}",          font=FONTS["body"], fill=(0,0,0)); y += 32
    draw.text((90, y),       f"Điều luật: {dieu_luat}",        font=FONTS["body"], fill=(0,0,0)); y += 32
    fields_gt["toi_danh"]   = (toi_danh,  (90+60, y-64, 90+60+500, y-36))
    fields_gt["dieu_luat"]  = (dieu_luat, (90+80, y-32, 90+80+500, y-4))
    y += 10

    # Quyết định
    y = _section_title(draw, "QUYẾT ĐỊNH:", 80, y)
    muc_ky = f"{random.randint(1,20)} năm tù"
    boi_thuong = f"{random.randint(10,500)*1_000_000:,} đồng".replace(",",".")
    qd_lines = [
        f"1. Tuyên bố bị cáo {bi_cao} phạm tội {toi_danh};",
        f"2. Xử phạt bị cáo {muc_ky} giam tại trại giam theo quy định pháp luật;",
        f"3. Bồi thường thiệt hại cho bị hại: {boi_thuong};",
        "4. Bị cáo có quyền kháng cáo bản án trong thời hạn 15 ngày kể từ ngày tuyên án.",
    ]
    for line in qd_lines:
        draw.text((90, y), line, font=FONTS["body"], fill=(0,0,0)); y += 32
    y += 20

    # Chữ ký
    _draw_signature(draw, 820, y, _rand_name(), title="TM. HỘI ĐỒNG XÉT XỬ\nTHẨM PHÁN CHỦ TỌA")
    _draw_seal(draw, 880, y + 160)
    _add_random_qr(img, f"CASE:{doc_id}:{so_vu.replace('/','-')}", prob=0.65)

    fields_gt["nguyen_don"] = (nguyen_don, (0, 0, 0, 0))   # không có trên trang này

    gt = _make_gt(doc_id, doc_type, "toa_an",
                  f"Bản án xét xử bị cáo {bi_cao} về tội {toi_danh}", fields_gt)
    return img, gt


def gen_bao_hiem() -> tuple:
    """Văn bản bảo hiểm: Hợp đồng, Thông báo bồi thường, Đơn yêu cầu."""
    doc_types = ["HỢP ĐỒNG BẢO HIỂM NHÂN THỌ", "THÔNG BÁO BỒI THƯỜNG BẢO HIỂM",
                 "ĐƠN YÊU CẦU BẢO HIỂM SỨC KHỎE", "BIÊN BẢN GIÁM ĐỊNH TỔN THẤT"]
    doc_type    = random.choice(doc_types)
    issuer      = random.choice(INSURANCE_ORGS)
    so_hd       = f"HD{random.randint(100000,999999)}/BH{random.randint(2020,2024)}"
    so_bhxh     = str(random.randint(10**9, 10**10 - 1))
    so_bhyt     = f"DN{random.randint(10**12, 10**13 - 1)}"
    ten_kh      = _rand_name()
    so_tien     = random.randint(5, 500) * 1_000_000
    doc_date    = _rand_date()
    doc_id      = uuid.uuid4().hex[:10].upper()

    img  = Image.new("RGB", (A4_W, A4_H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    fields_gt = {}

    # Header
    draw.text((80, 60), issuer, font=FONTS["subtitle"], fill=(0, 0, 100))
    draw.text((80, 100), "Chi nhánh: TP. Hồ Chí Minh", font=FONTS["body"], fill=(0,0,0))
    draw.text((80, 130), f"Số HĐ: {so_hd}", font=FONTS["label"], fill=(0,0,0))
    draw.text((640, 60), "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
              font=FONTS["body"], fill=(0,0,0))
    draw.text((730, 98), "Độc lập - Tự do - Hạnh phúc",
              font=FONTS["body"], fill=(0,0,0))
    draw.line((730, 128, 1160, 128), fill=(0,0,0), width=2)
    draw.text((800, 140), f"TP. Hồ Chí Minh, {doc_date}", font=FONTS["small"], fill=(0,0,0))
    draw.line((80, 180, 1160, 180), fill=(0,0,0), width=2)

    fields_gt["so_hop_dong"]      = (so_hd,    (80, 130, 80+300, 158))
    fields_gt["ngay_thang_nam"]   = (doc_date, (800, 140, 1160, 168))
    fields_gt["co_quan_ban_hanh"] = (issuer,   (80, 60, 640, 88))

    y = 200
    tw = draw.textlength(doc_type, FONTS["title"])
    draw.text(((A4_W - tw) // 2, y), doc_type, font=FONTS["title"], fill=(0,0,0))
    fields_gt["ten_loai_van_ban"] = (doc_type, ((A4_W-tw)//2, y, (A4_W+tw)//2, y+45))
    y += 70

    # Thông tin khách hàng – dạng biểu mẫu (handwritten = True giả lập điền tay)
    y = _section_title(draw, "THÔNG TIN NGƯỜI THAM GIA BẢO HIỂM:", 80, y)
    kh_data = [
        ("Họ và tên:",          ten_kh,       True),
        ("Ngày sinh:",          f"{random.randint(1,28)}/{random.randint(1,12)}/{random.randint(1960,2005)}", True),
        ("Số sổ BHXH:",         so_bhxh,      True),
        ("Số thẻ BHYT:",        so_bhyt,      True),
        ("CCCD/CMND:",          _rand_cccd(), True),
        ("Địa chỉ:",            _rand_address(), True),
        ("Điện thoại liên hệ:", _rand_phone(), True),
        ("Email:",              f"user{random.randint(100,999)}@gmail.com", False),
    ]
    for lbl, val, hw in kh_data:
        bb = _form_field(draw, lbl, val, 80, y, 420, 1150, handwritten=hw)
        if "Sổ BHXH" in lbl:
            fields_gt["so_so_bhxh"] = (val, bb)
        elif "BHYT" in lbl:
            fields_gt["so_the_bhyt"] = (val, bb)
        elif "tên" in lbl:
            fields_gt["ten_nguoi_tham_gia"] = (val, bb)
        y += 42
    y += 15

    # Bảng quyền lợi
    y = _section_title(draw, "QUYỀN LỢI BẢO HIỂM:", 80, y)
    headers   = ["STT", "Quyền lợi bảo hiểm",        "Số tiền BH (VNĐ)",         "Thời hạn"]
    col_widths = [55,    450,                           320,                         175]
    rows = [
        ["1", "Tử vong / Thương tật toàn bộ vĩnh viễn", f"{so_tien:,}".replace(",","."), "20 năm"],
        ["2", "Bệnh hiểm nghèo (30 loại bệnh)",          f"{so_tien//2:,}".replace(",","."), "20 năm"],
        ["3", "Trợ cấp nằm viện hàng ngày",              f"{random.randint(200,500)*1000:,}".replace(",",".")+" /ngày", "20 năm"],
    ]
    y = _draw_table(draw, 80, y, headers, rows, col_widths) + 20
    fields_gt["so_tien_boi_thuong"] = (f"{so_tien:,}", (80, y - 3*36 - 36, 80+col_widths[0]+col_widths[1]+col_widths[2], y - 36))

    # Điều khoản
    draw.text((80, y), f"Phí bảo hiểm định kỳ: {random.randint(2,10)*1_000_000:,} VNĐ/{random.choice(['tháng','quý','năm'])}".replace(",","."),
              font=FONTS["body"], fill=(0,0,0)); y += 35
    draw.text((80, y), f"Ngày hiệu lực hợp đồng: {doc_date}", font=FONTS["body"], fill=(0,0,0)); y += 50

    # Chữ ký hai bên
    draw.text((80,  y), "BÊN MUA BẢO HIỂM", font=FONTS["label"], fill=(0,0,0))
    _draw_signature(draw, 80, y+30, ten_kh)
    draw.text((800, y), "ĐẠI DIỆN CÔNG TY BH", font=FONTS["label"], fill=(0,0,0))
    _draw_signature(draw, 800, y+30, _rand_name())
    _draw_seal(draw, 870, y+170)

    _add_random_qr(img, f"BH:{doc_id}:{so_hd}", prob=0.65)

    gt = _make_gt(doc_id, doc_type, "bao_hiem",
                  f"Hợp đồng bảo hiểm nhân thọ cho khách hàng {ten_kh}", fields_gt)
    return img, gt


def gen_cong_an() -> tuple:
    """Văn bản Công an: Biên bản vi phạm, Quyết định xử phạt, Lệnh khám xét."""
    doc_types = ["BIÊN BẢN VI PHẠM HÀNH CHÍNH",
                 "QUYẾT ĐỊNH XỬ PHẠT VI PHẠM HÀNH CHÍNH",
                 "LỆNH KHÁM XÉT",
                 "THÔNG BÁO TẠM HOÃN XUẤT CẢNH"]
    doc_type  = random.choice(doc_types)
    don_vi    = random.choice(["ĐỘI CSGT SỐ 1", "ĐỘI CSGT SỐ 5", "CA PHƯỜNG XYZ",
                                "BAN CSĐT CA QUẬN"])
    doc_no    = _rand_doc_no("BB", random.choice(["CSGT","CA","CSHS"]))
    doc_date  = _rand_date()
    doc_id    = uuid.uuid4().hex[:10].upper()

    ten_vp    = _rand_name()
    cccd_vp   = _rand_cccd()
    bien_so   = _rand_bien_so()
    vp_info   = random.choice(VI_PHAM_GTDB)

    img  = Image.new("RGB", (A4_W, A4_H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    fields_gt = {}

    y = _header_block(draw, f"CÔNG AN TP. HỒ CHÍ MINH", don_vi,
                      doc_no, f"TP. Hồ Chí Minh, {doc_date}")
    fields_gt["so_van_ban"]       = (doc_no,   (80, 132, 80+300, 160))
    fields_gt["ngay_thang_nam"]   = (doc_date, (800,138, 1160,  166))
    fields_gt["co_quan_ban_hanh"] = (don_vi,   (80, 98,  80+450, 126))

    y = _doc_title_block(draw, doc_type, "Vi phạm hành chính trong lĩnh vực GTĐB", y)
    fields_gt["ten_loai_van_ban"] = (doc_type, (300, y-107, 1000, y-55))
    y += 10

    # Người vi phạm
    y = _section_title(draw, "NGƯỜI VI PHẠM:", 80, y)
    vp_fields = [
        ("Họ và tên:",          ten_vp,           True),
        ("Ngày sinh:",          f"{random.randint(1,28)}/{random.randint(1,12)}/{random.randint(1970,2005)}", True),
        ("CCCD/CMND số:",       cccd_vp,          True),
        ("Địa chỉ thường trú:", _rand_address(),  True),
        ("GPLX số:",            _rand_cccd(),     True),
        ("Hạng GPLX:",          random.choice(LICENSE_CLASSES), True),
    ]
    for lbl, val, hw in vp_fields:
        bb = _form_field(draw, lbl, val, 90, y, 380, 1150, handwritten=hw)
        if "tên" in lbl:
            fields_gt["ten_nguoi_vi_pham"] = (val, bb)
        elif "CCCD" in lbl:
            fields_gt["cmnd_cccd"] = (val, bb)
        y += 38
    y += 10

    # Phương tiện
    y = _section_title(draw, "PHƯƠNG TIỆN:", 80, y)
    pt_fields = [
        ("Biển số xe:",  bien_so,                             True),
        ("Loại xe:",     random.choice(VEHICLE_TYPES),        False),
        ("Nhãn hiệu:",   random.choice(VEHICLE_BRANDS),       False),
        ("Màu sơn:",     random.choice(["Trắng","Đen","Đỏ","Xanh","Bạc"]), False),
    ]
    for lbl, val, hw in pt_fields:
        bb = _form_field(draw, lbl, val, 90, y, 380, 900, handwritten=hw)
        if "Biển số" in lbl:
            fields_gt["bien_so_xe"] = (val, bb)
        y += 38
    y += 10

    # Hành vi vi phạm
    y = _section_title(draw, "HÀNH VI VI PHẠM:", 80, y)
    draw.text((90, y), f"Hành vi:  {vp_info[0]}", font=FONTS["body"],  fill=(0,0,0)); y += 32
    draw.text((90, y), f"Điều luật: {vp_info[1]}", font=FONTS["body"], fill=(0,0,0)); y += 32
    draw.text((90, y), f"Mức phạt:  {vp_info[2]} VNĐ", font=FONTS["label"], fill=(160,0,0)); y += 32
    fields_gt["hanh_vi_vi_pham"] = (vp_info[0], (90+80,  y-96, 90+80+500, y-64))
    fields_gt["dieu_luat"]       = (vp_info[1], (90+80,  y-64, 90+80+600, y-32))
    fields_gt["muc_phat"]        = (vp_info[2], (90+80,  y-32, 90+80+200, y))

    loc_labels = ["Ngã tư Lê Lợi - Nguyễn Huệ", f"Km {random.randint(1,50)}+{random.randint(100,900)} QL{random.randint(1,22)}A",
                  f"Đường {random.choice(STREETS)}, {random.choice(DISTRICTS)}"]
    y += 10
    draw.text((90, y), f"Địa điểm: {random.choice(loc_labels)}", font=FONTS["body"], fill=(0,0,0)); y += 32
    draw.text((90, y), f"Thời gian: {random.randint(6,23)}h{random.randint(0,59):02d}", font=FONTS["body"], fill=(0,0,0)); y += 45

    # Chữ ký
    draw.text((80,  y), "NGƯỜI VI PHẠM", font=FONTS["label"], fill=(0,0,0))
    _draw_signature(draw, 80, y+30, ten_vp)
    draw.text((820, y), "CBCS LẬP BIÊN BẢN", font=FONTS["label"], fill=(0,0,0))
    _draw_signature(draw, 820, y+30, _rand_name())
    _draw_seal(draw, 880, y+160)

    _add_random_qr(img, f"CA:{doc_id}:{doc_no.replace('/','-')}", prob=0.65)

    gt = _make_gt(doc_id, doc_type, "cong_an",
                  f"Biên bản vi phạm hành chính lĩnh vực GTĐB, xe {bien_so}", fields_gt)
    return img, gt


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 – IMAGE DEGRADATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DEGRAD_CONFIG = {
    "yellowing":    0.30,  # ố vàng / sepia
    "gaussian":     0.40,  # Gaussian noise
    "salt_pepper":  0.20,  # nhiễu muối-tiêu
    "upside_down":  0.10,  # ngược chiều 180°  (Module 1.2)
    "skew":         0.45,  # nghiêng góc       (Module 1.1)
    "black_border": 0.35,  # viền đen máy scan (Module 1.1)
    "blur":         0.30,  # nhòe / mờ
    "fold":         0.25,  # nếp gấp
    "shadow":       0.20,  # bóng scan góc cạnh
    "faded":        0.30,  # chữ phai nhạt     (Module 1.1)
}


def apply_degradations(img_pil: Image.Image,
                       cfg: dict = None) -> tuple:
    """
    Áp dụng ngẫu nhiên các kỹ thuật làm hỏng ảnh (Module 1).
    Trả về (img_cv_BGR, list[str] degradation tags).
    """
    if cfg is None:
        cfg = DEFAULT_DEGRAD_CONFIG

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    tags = []

    # 1. Yellowing / Sepia
    if random.random() < cfg.get("yellowing", 0):
        M = np.array([[0.272, 0.534, 0.131],
                      [0.349, 0.686, 0.168],
                      [0.393, 0.769, 0.189]])
        f = img.astype(np.float64)
        s = np.stack([np.sum(f * M[i], axis=2) for i in range(3)], axis=2)
        img = np.clip(s, 0, 255).astype(np.uint8)
        tags.append("yellowed")

    # 2. Gaussian noise
    if random.random() < cfg.get("gaussian", 0):
        std = random.uniform(8, 28)
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        tags.append(f"gnoise{int(std)}")

    # 3. Salt & Pepper
    if random.random() < cfg.get("salt_pepper", 0):
        amount = random.uniform(0.002, 0.012)
        n = int(img.size * amount)
        for _ in range(n):
            r, c = random.randint(0, img.shape[0]-1), random.randint(0, img.shape[1]-1)
            img[r, c] = 255 if random.random() < 0.5 else 0
        tags.append("salt_pepper")

    # 4. Upside-down (Module 1.2 – phát hiện ảnh ngược)
    if random.random() < cfg.get("upside_down", 0):
        img = cv2.rotate(img, cv2.ROTATE_180)
        tags.append("upside_down")

    # 5. Skew / Deskew test (Module 1.1)
    if random.random() < cfg.get("skew", 0):
        angle = random.uniform(-15.0, 15.0)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
        tags.append(f"skew{round(angle,1)}")

    # 6. Black scan border (Module 1.1 – auto-crop)
    if random.random() < cfg.get("black_border", 0):
        t = random.randint(20, 90);  b = random.randint(20, 90)
        l = random.randint(20, 90);  r = random.randint(20, 90)
        img = cv2.copyMakeBorder(img, t, b, l, r,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
        tags.append("black_border")

    # 7. Blur (nhòe máy scan)
    if random.random() < cfg.get("blur", 0):
        k = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)
        tags.append(f"blur{k}")

    # 8. Fold lines (nếp gấp)
    if random.random() < cfg.get("fold", 0):
        h, w = img.shape[:2]
        for _ in range(random.randint(1, 3)):
            axis = random.choice(["v", "h"])
            alpha = random.uniform(0.65, 0.88)
            if axis == "v":
                x = random.randint(w // 5, 4 * w // 5)
                sw = random.randint(4, 18)
                img[:, max(0,x-sw//2):min(w,x+sw//2)] = \
                    (img[:, max(0,x-sw//2):min(w,x+sw//2)] * alpha).astype(np.uint8)
            else:
                y = random.randint(h // 5, 4 * h // 5)
                sh = random.randint(4, 18)
                img[max(0,y-sh//2):min(h,y+sh//2), :] = \
                    (img[max(0,y-sh//2):min(h,y+sh//2), :] * alpha).astype(np.uint8)
        tags.append("fold")

    # 9. Edge shadow (bóng scan, Module 1.1)
    if random.random() < cfg.get("shadow", 0):
        h, w = img.shape[:2]
        edge  = random.choice(["top", "bottom", "left", "right"])
        depth = random.randint(60, 220)
        alpha = random.uniform(0.55, 0.82)
        mask  = np.ones((h, w), dtype=np.float32)
        if   edge == "top":    mask[:depth, :]  = alpha
        elif edge == "bottom": mask[h-depth:, :] = alpha
        elif edge == "left":   mask[:, :depth]  = alpha
        else:                  mask[:, w-depth:] = alpha
        img = (img * mask[:, :, np.newaxis]).astype(np.uint8)
        tags.append(f"shadow_{edge}")

    # 10. Faded text (chữ phai, Module 1.1)
    if random.random() < cfg.get("faded", 0):
        alpha = random.uniform(0.60, 0.85)
        white = (np.ones_like(img) * 255).astype(np.uint8)
        img = cv2.addWeighted(img, alpha, white, 1 - alpha, 0)
        tags.append("faded")

    return img, tags


def make_blank_page() -> np.ndarray:
    """Trang trắng máy scan (Module 1.2 – phát hiện trang trắng)."""
    img = np.full((A4_H, A4_W, 3), 252, dtype=np.uint8)
    noise = np.random.normal(0, random.uniform(3, 10), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.5:
        border = random.randint(10, 45)
        img[:border, :] = 0
        img[-border:, :] = 0
    return img


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 – PDF BUNDLE GENERATOR (PDF Logical Splitting test)
# ─────────────────────────────────────────────────────────────────────────────

def generate_bundle(bundle_idx: int, pdf_dir: str,
                    docs_per_bundle=(3, 7),
                    pages_per_doc=(1, 4)) -> dict:
    """
    Tạo một multi-document bundle (TIFF multi-page hoặc PDF) cùng manifest JSON.
    Manifest chuẩn API contract Module 5 (sub_documents).
    """
    bundle_id = f"BUNDLE_{bundle_idx:02d}_{uuid.uuid4().hex[:6].upper()}"
    pages_pil  = []
    manifest   = {"bundle_id": bundle_id, "sub_documents": []}
    page_ptr   = 1

    generators = [gen_hanh_chinh, gen_toa_an, gen_bao_hiem, gen_cong_an]
    n_docs = random.randint(*docs_per_bundle)

    bundle_cfg = {k: v * 0.6 for k, v in DEFAULT_DEGRAD_CONFIG.items()}
    bundle_cfg["upside_down"] = 0.0   # không nghiêng trong bundle

    for _ in range(n_docs):
        img_pil, gt = random.choice(generators)()
        img_cv, _   = apply_degradations(img_pil, cfg=bundle_cfg)
        page_img    = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        n_pages = random.randint(*pages_per_doc)
        start_p = page_ptr
        for p in range(n_pages):
            if p == 0:
                pages_pil.append(page_img)
            else:
                # Trang phụ: trang trắng hoặc bản sao mờ
                if random.random() < 0.3:
                    blank = Image.fromarray(make_blank_page())
                    pages_pil.append(blank)
                else:
                    pages_pil.append(page_img)
            page_ptr += 1

        manifest["sub_documents"].append({
            "type":           gt["classification"],
            "sector":         gt["sector"],
            "page_start":     start_p,
            "page_end":       page_ptr - 1,
            "summary":        gt["summary"],
            "expected_fields": {k: v["value"] if isinstance(v, dict) else v
                                for k, v in [
                                    (m["field_name"], m["value"])
                                    for m in gt["metadata"]
                                ]},
        })

    if not pages_pil:
        return {}

    # Lưu multi-page TIFF (luôn được)
    tiff_path = os.path.join(pdf_dir, f"{bundle_id}.tiff")
    pages_pil[0].save(tiff_path, save_all=True,
                      append_images=pages_pil[1:],
                      compression="tiff_deflate")

    # Thử lưu PDF
    try:
        pdf_path = os.path.join(pdf_dir, f"{bundle_id}.pdf")
        pages_pil[0].save(pdf_path, save_all=True, append_images=pages_pil[1:])
        manifest["files"] = {"tiff": f"{bundle_id}.tiff", "pdf": f"{bundle_id}.pdf"}
    except Exception:
        manifest["files"] = {"tiff": f"{bundle_id}.tiff"}

    manifest["total_pages"] = page_ptr - 1
    manifest["total_docs"]  = n_docs

    man_path = os.path.join(pdf_dir, f"{bundle_id}_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_GEN_MAP = {
    "hanh_chinh": gen_hanh_chinh,
    "toa_an":     gen_toa_an,
    "bao_hiem":   gen_bao_hiem,
    "cong_an":    gen_cong_an,
}


def run(
    output_dir:        str   = "dataset_vn_digitize",
    num_images:        int   = 60,
    blank_ratio:       float = 0.08,
    num_bundles:       int   = 5,
    sector_weights:    dict  = None,
    degrad_config:     dict  = None,
    jpeg_quality:      int   = 88,
):
    """
    Pipeline chính.

    Args:
        output_dir      : Thư mục gốc chứa toàn bộ output
        num_images      : Số ảnh đơn cần sinh (Module 1/2/3)
        blank_ratio     : Tỉ lệ trang trắng 0–1 (Module 1.2)
        num_bundles     : Số PDF bundle cần sinh (Module 4)
        sector_weights  : {'hanh_chinh':2,'toa_an':1,...} – tỉ lệ sinh
        degrad_config   : Ghi đè xác suất từng degradation
        jpeg_quality    : Chất lượng JPEG output (50–100)
    """
    img_dir  = os.path.join(output_dir, "images")
    gt_dir   = os.path.join(output_dir, "ground_truth")
    pdf_dir  = os.path.join(output_dir, "pdf_bundles")
    for d in (img_dir, gt_dir, pdf_dir):
        os.makedirs(d, exist_ok=True)

    if sector_weights is None:
        sector_weights = {"hanh_chinh": 3, "toa_an": 2, "bao_hiem": 2, "cong_an": 2}
    sectors_pool = [s for s, w in sector_weights.items() for _ in range(w)]

    cfg = degrad_config or DEFAULT_DEGRAD_CONFIG

    stats = {
        "total": 0, "blank": 0,
        "by_sector": {s: 0 for s in SECTOR_GEN_MAP},
        "clean": 0,
    }
    
    csv_data = [] # Lưu trữ dữ liệu để ghi ra file CSV labels

    print("=" * 65)
    print(" VN-Digitize Test Data Generator  –  AI Requirements v2.0")
    print("=" * 65)
    print(f" Output dir : {output_dir}")
    print(f" Images     : {num_images}  (blank ≈{int(blank_ratio*100)}%)")
    print(f" PDF bundles: {num_bundles}")
    print("-" * 65)

    # ── Single images ──────────────────────────────────────────────
    for i in range(num_images):
        prefix = f"img_{i:04d}"

        # Blank page
        if random.random() < blank_ratio:
            img_cv = make_blank_page()
            fname  = f"{prefix}_BLANK.jpg"
            cv2.imwrite(os.path.join(img_dir, fname),
                        img_cv, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            gt = {"document_id": f"BLANK_{i:04d}", "sector": "blank",
                  "classification": "BLANK_PAGE",
                  "summary": "Trang trắng – không có nội dung",
                  "confidence_overall": 0.0, "metadata": [], "sub_documents": [],
                  "image_file": fname}
            with open(os.path.join(gt_dir, f"{prefix}_BLANK.json"), "w",
                      encoding="utf-8") as f:
                json.dump(gt, f, ensure_ascii=False, indent=2)
            
            # Lưu log dòng blank vào CSV
            csv_data.append([
                fname, f"{prefix}_BLANK.json", gt["document_id"], 
                gt["sector"], gt["classification"], "blank_page"
            ])

            stats["blank"] += 1
            stats["total"] += 1
            print(f"  [{i+1:3d}/{num_images}] BLANK  → {fname}")
            continue

        # Normal document
        sector  = random.choice(sectors_pool)
        img_pil, gt = SECTOR_GEN_MAP[sector]()
        img_cv, tags = apply_degradations(img_pil, cfg=cfg)

        tag_str = "_".join(tags[:4]) if tags else "clean"
        fname   = f"{prefix}_{sector}_{tag_str}.jpg"
        cv2.imwrite(os.path.join(img_dir, fname),
                    img_cv, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        gt["degradations"] = tags
        gt["image_file"]   = fname
        with open(os.path.join(gt_dir, f"{prefix}_{sector}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)

        # Lưu log dòng thông thường vào CSV
        csv_data.append([
            fname, f"{prefix}_{sector}.json", gt["document_id"], 
            gt["sector"], gt["classification"], ",".join(tags) if tags else "clean"
        ])

        stats["total"] += 1
        stats["by_sector"][sector] += 1
        if not tags:
            stats["clean"] += 1
        print(f"  [{i+1:3d}/{num_images}] {sector:<14} → {fname[:62]}")

    # ── PDF bundles (Module 4) ──────────────────────────────────────
    print(f"\n  Generating {num_bundles} PDF bundle(s) for Module 4 testing…")
    bundle_stats = []
    for b in range(num_bundles):
        m = generate_bundle(b + 1, pdf_dir)
        if m:
            bundle_stats.append(m)
            files_str = " / ".join(m.get("files", {}).values())
            print(f"  Bundle {b+1:02d}: {m['total_pages']} pages, "
                  f"{m['total_docs']} docs  → {files_str}")

    # Ghi file CSV Labels
    csv_path = os.path.join(output_dir, "dataset_labels.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "ground_truth_filename", "document_id", "sector", "classification", "degradations"])
        writer.writerows(csv_data)

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" HOÀN TẤT!")
    print(f"  Tổng ảnh       : {stats['total']}")
    print(f"  Trang trắng    : {stats['blank']}")
    print(f"  Ảnh sạch (clean): {stats['clean']}")
    print(f"  Theo sector    : {stats['by_sector']}")
    print(f"  PDF bundles    : {len(bundle_stats)}")
    print(f"\n  📁 {img_dir}")
    print(f"  📁 {gt_dir}")
    print(f"  📁 {pdf_dir}")
    print(f"  📄 {csv_path}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        output_dir    = "dataset_vn_digitize",
        num_images    = 60,           # ảnh đơn (Module 1/2/3)
        blank_ratio   = 0.08,         # ~8% trang trắng
        num_bundles   = 5,            # PDF bundle (Module 4)
        sector_weights= {             # tỉ lệ sinh theo nghiệp vụ
            "hanh_chinh": 3,
            "toa_an":     2,
            "bao_hiem":   2,
            "cong_an":    2,
        },
        jpeg_quality  = 88,
    )
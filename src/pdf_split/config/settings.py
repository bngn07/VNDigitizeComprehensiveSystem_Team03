"""
Toàn bộ cấu hình của Module 4.1 — chỉnh sửa tại đây, không cần đụng code logic.
"""
import re
from pathlib import Path

# ─── Đường dẫn ────────────────────────────────────────────────────────────────

BASE_DIR    = Path(r"D:\VNDigitizeComprehensiveSystem_Team03\data") 
PDF_PATH    = BASE_DIR / "input" / "document_merged2.pdf"
OUTPUT_DIR  = BASE_DIR / "output"
CACHE_PATH  = BASE_DIR / "cache_features.pkl"

# ─── OCR & Trích xuất ─────────────────────────────────────────────────────────
CACHE_VERSION  = "7.0"    # tăng số này để vô hiệu hoá cache cũ
OCR_DPI        = 200      # DPI render trang scan trước khi OCR
BLANK_THRESH   = 30       # số ký tự tối thiểu để coi trang không phải blank
TEXT_MIN_CHARS = 20       # số ký tự (sau bỏ khoảng trắng) để xác định trang có text layer
OCR_WORKERS    = 4        # số process song song cho Tesseract

# ─── Phân đoạn ────────────────────────────────────────────────────────────────
MIN_SEG_PAGES = 1         # segment có ≤ N trang không có dấu hiệu rõ → merge vào trước

# ─── Phân loại ────────────────────────────────────────────────────────────────
LABEL_SET = {"Quyết định", "Bản án dân sự", "Bản án hình sự", "Bản án hành chính"}

# ─── Qwen verifier (tuỳ chọn) ─────────────────────────────────────────────────
QWEN_URL   = "http://localhost:11434/api/generate"
QWEN_MODEL = "qwen2.5:7b"
QWEN_TO    = 30           # timeout giây

# ─── Ground truth (dùng để evaluate) ─────────────────────────────────────────
# key = trang bắt đầu của segment, value = nhãn đúng
GT_SEGMENT_TYPES: dict[int, str] = {
     1: "Quyết định",      4: "Quyết định",      7: "Quyết định",
     9: "Bản án dân sự",  29: "Quyết định",     35: "Bản án dân sự",
    43: "Bản án hình sự", 44: "Bản án hình sự",  50: "Quyết định",
    52: "Quyết định",     54: "Quyết định",      57: "Quyết định",
    59: "Quyết định",     62: "Quyết định",      63: "Quyết định",
    66: "Bản án hình sự", 75: "Bản án dân sự",   81: "Bản án hành chính",
   103: "Quyết định",    105: "Quyết định",     108: "Quyết định",
}

# ─── Regex dùng chung ─────────────────────────────────────────────────────────
RE_SO_VAN_BAN = re.compile(r'\d{1,3}/\d{4,}/[A-ZĐQĐ][\w\-]+', re.IGNORECASE)

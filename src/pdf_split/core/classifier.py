"""
Phân loại loại văn bản pháp lý Việt Nam theo 4 nhãn:
    Quyết định | Bản án dân sự | Bản án hình sự | Bản án hành chính

Chiến lược 4 lớp (ưu tiên từ cao xuống thấp):
    1. Dấu hiệu cứng (QUYẾT ĐỊNH + động từ, ĐÌNH CHỈ ...)
    2. Ký hiệu số văn bản  (/QĐ-, /HC-ST, /HS-PT, /DS-ST ...)
    3. Tính điểm từ khoá trên toàn văn bản
    4. Qwen verifier (LLM địa phương, fallback khi rule < 0.80)
"""
import re
import requests

from config.settings import LABEL_SET, QWEN_URL, QWEN_MODEL, QWEN_TO, RE_SO_VAN_BAN


# ─── Lớp 1: Dấu hiệu cứng ────────────────────────────────────────────────────

_RE_HARD_QUYET_DINH = re.compile(
    r'QUYẾT ĐỊNH\s*(ĐÌNH CHỈ|V/V|SỐ\s*\d|XÉT XỬ|CÔNG NHẬN)', re.I
)
_DINH_CHI_PHRASES = ("ĐÌNH CHỈ VỤ ÁN", "ĐÌNH CHỈ XÉT XỬ", "ĐÌNH CHỈ PHIÊN TÒA")


# ─── Lớp 2: Ký hiệu số văn bản ───────────────────────────────────────────────

# Tra trực tiếp chuỗi (ưu tiên cao nhất)
_SPECIAL_SUFFIX_MAP: dict[str, str] = {
    'QĐST-HC': "Quyết định",
    'QĐPT-HC': "Quyết định",
    'QĐ-PT'  : "Quyết định",
    'QĐ-ST'  : "Quyết định",
}

# Regex ký hiệu phần sau dấu /
_KY_HIEU_PATTERNS: list[tuple] = [
    (re.compile(r'/QĐ[\-\w]*|/QĐST[\-\w]*|/QĐPT[\-\w]*', re.I), "Quyết định"),
    (re.compile(r'/[\w]*HC[\-]*(ST|PT|GĐT)\b',               re.I), "Bản án hành chính"),
    (re.compile(r'/[\w]*HS[\-]*(ST|PT|GĐT|SĐT)\b',           re.I), "Bản án hình sự"),
    (re.compile(r'/[\w]*(DS|HNGĐ|KDTM)[\-]*(ST|PT|GĐT)?\b',  re.I), "Bản án dân sự"),
]


def _classify_by_so_van_ban(so_van_ban: str) -> str | None:
    su = so_van_ban.upper()
    for suffix, label in _SPECIAL_SUFFIX_MAP.items():
        if suffix in su:
            return label
    for pattern, label in _KY_HIEU_PATTERNS:
        if pattern.search(so_van_ban):
            return label
    return None


# ─── Lớp 3: Tính điểm từ khoá ────────────────────────────────────────────────

_KEYWORD_SCORES: list[tuple] = [
    ("BẢN ÁN HÌNH SỰ",               "Bản án hình sự",   5),
    ("BỊ CÁO",                        "Bản án hình sự",   4),
    ("VỀ TỘI",                        "Bản án hình sự",   3),
    ("PHẠM TỘI",                      "Bản án hình sự",   3),
    ("BẢN ÁN DÂN SỰ",                "Bản án dân sự",    5),
    ("BẢN ÁN HÔN NHÂN",              "Bản án dân sự",    5),
    ("LY HÔN",                        "Bản án dân sự",    4),
    ("TRANH CHẤP HỢP ĐỒNG",          "Bản án dân sự",    3),
    ("NGUYÊN ĐƠN",                    "Bản án dân sự",    2),
    ("BỊ ĐƠN",                        "Bản án dân sự",    2),
    ("BẢN ÁN HÀNH CHÍNH",            "Bản án hành chính", 5),
    ("KHỞI KIỆN QUYẾT ĐỊNH HÀNH CHÍNH", "Bản án hành chính", 5),
    ("NGƯỜI BỊ KIỆN",                "Bản án hành chính", 3),
]


def _score_keywords(text: str) -> tuple[str | None, float]:
    """Tính điểm từ khoá, trả về (nhãn tốt nhất, confidence)."""
    t   = text[:2000].upper()
    t5  = text[:500].upper()
    scores = dict.fromkeys(LABEL_SET, 0)

    # Ưu tiên "Quyết định" khi dấu hiệu cứng xuất hiện trong 500 ký tự đầu
    is_quyet_dinh = (
        bool(re.search(r'QUYẾT ĐỊNH\s*[\n\r]', t5))
        or bool(_RE_HARD_QUYET_DINH.search(t5))
        or any(p in t5 for p in _DINH_CHI_PHRASES)
    )

    if is_quyet_dinh:
        scores["Quyết định"] += 5
        for kw, label, weight in _KEYWORD_SCORES:
            if label not in ("Bản án hình sự", "Bản án dân sự") and kw in t:
                scores[label] += weight
    else:
        if re.search(r'QUYẾT ĐỊNH\s{0,5}(V/V|SỐ|ĐÌNH CHỈ)', t5):
            scores["Quyết định"] += 5
        for kw, label, weight in _KEYWORD_SCORES:
            if kw in t:
                scores[label] += weight

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None, 0.0

    vals = sorted(scores.values(), reverse=True)
    confidence = min(0.95, 0.6 + (vals[0] - vals[1]) * 0.05)
    return best, confidence


# ─── Lớp 4: Qwen verifier ────────────────────────────────────────────────────

_QWEN_PROMPT = (
    "Phân loại tài liệu pháp lý Việt Nam sau vào đúng 1 nhãn "
    "(chỉ trả nhãn, không giải thích):\n"
    "Quyết định | Bản án dân sự | Bản án hình sự | Bản án hành chính"
)


def qwen_classify(text: str) -> tuple[str | None, float]:
    """Gọi Qwen qua Ollama để verify. Trả về (nhãn, confidence) hoặc (None, 0)."""
    try:
        resp = requests.post(
            QWEN_URL, timeout=QWEN_TO,
            json={
                "model"  : QWEN_MODEL,
                "stream" : False,
                "prompt" : f"{_QWEN_PROMPT}\n\n{text[:800]}\n\nNhãn:",
                "options": {"temperature": 0},
            }
        )
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip()
            for label in LABEL_SET:
                if label.lower() in raw.lower():
                    return label, 0.88
    except Exception:
        pass
    return None, 0.0


def qwen_is_available() -> bool:
    try:
        return requests.get("http://localhost:11434", timeout=3).status_code == 200
    except Exception:
        return False


# ─── API chính ────────────────────────────────────────────────────────────────

def classify_rule(pages: list[dict]) -> tuple[str, float]:
    """
    Phân loại bằng rule-based (không cần LLM).

    Nhận vào danh sách trang (dict feature) của 1 segment.
    Trả về (nhãn, confidence).
    """
    texts  = [p["text_full"] for p in pages[:3]]
    header = texts[0][:500].upper() if texts else ""

    # Lớp 1: dấu hiệu cứng
    if _RE_HARD_QUYET_DINH.search(header) or any(p in header for p in _DINH_CHI_PHRASES):
        return "Quyết định", 0.97

    # Lớp 2: ký hiệu số văn bản từ các trang đầu
    for text in texts:
        for so_vb in RE_SO_VAN_BAN.findall(text):
            label = _classify_by_so_van_ban(so_vb)
            if label:
                return label, 0.98

    # Lớp 3: tính điểm từ khoá
    label, conf = _score_keywords(" ".join(texts))
    if label:
        return label, conf

    # Lớp 3b: thử sang trang 2 nếu trang 1 không đủ thông tin
    if len(pages) > 1:
        for so_vb in RE_SO_VAN_BAN.findall(pages[1]["text_full"]):
            label = _classify_by_so_van_ban(so_vb)
            if label:
                return label, 0.90
        label2, conf2 = _score_keywords(pages[1]["text_full"])
        if label2:
            return label2, max(0.5, conf2 - 0.1)

    # Fallback
    return "Quyết định", 0.50

"""
Trích xuất văn bản từ PDF:
- Trang có text layer → dùng trực tiếp (PyMuPDF)
- Trang scan / không có text → render pixmap rồi OCR (Tesseract)
- Kết quả được cache theo md5 để tránh extract lại khi file không đổi
"""
import re
import time
import pickle
import hashlib
import concurrent.futures
from pathlib import Path

import fitz
from PIL import Image
import pytesseract
from tqdm.auto import tqdm

from config.settings import (
    CACHE_VERSION, OCR_DPI, BLANK_THRESH, TEXT_MIN_CHARS, OCR_WORKERS,
    RE_SO_VAN_BAN,
)


# ─── Cache ────────────────────────────────────────────────────────────────────

def _cache_key(pdf_path: str) -> str:
    mtime = str(Path(pdf_path).stat().st_mtime) if Path(pdf_path).exists() else "0"
    return hashlib.md5(f"{pdf_path}|{mtime}|{CACHE_VERSION}".encode()).hexdigest()[:12]


def _load_cache(cache_path: str, pdf_path: str):
    if not Path(cache_path).exists():
        return None
    try:
        bundle = pickle.load(open(cache_path, "rb"))
        if bundle.get("key") != _cache_key(pdf_path):
            print("  ⚠️  Cache lỗi thời → extract lại")
            return None
        return bundle["features"]
    except Exception:
        return None


def _save_cache(cache_path: str, pdf_path: str, features: list):
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"key": _cache_key(pdf_path), "features": features}, open(cache_path, "wb"))


# ─── Xử lý header / đặc trưng từng trang ─────────────────────────────────────

def _strict_header(text: str, n: int = 7) -> tuple[str, str]:
    """Lấy n dòng đầu tiên có nội dung thực sự (bỏ dòng trống và số trang)."""
    result, first = [], ""
    for line in (text.split("\n") if text else []):
        s = line.strip()
        if not s or re.match(r'^[\d\-–—\s]{1,5}$', s):
            continue
        if not first:
            first = s
        result.append(s)
        if len(result) >= n:
            break
    return " ".join(result), first


def _build_features(page_idx: int, text: str) -> dict:
    """Chuyển text thô của 1 trang thành dict đặc trưng dùng cho boundary & classifier."""
    header = " ".join(
        l.strip() for l in text[:2000].split("\n")[:10] if l.strip()
    )
    header_upper = header.upper()
    strict_hdr   = _strict_header(text)

    return {
        "page_num"      : page_idx + 1,
        "is_blank"      : len(text) < BLANK_THRESH,
        "header"        : header,
        "has_quoc_hieu" : "CỘNG HÒA XÃ HỘI" in header_upper,
        "has_toa_an"    : "TÒA ÁN NHÂN DÂN"  in header_upper,
        "so_vb_header"  : RE_SO_VAN_BAN.findall(header),
        "text_full"     : text,
        "strict_header" : strict_hdr,
    }


# ─── OCR worker (chạy trong subprocess) ───────────────────────────────────────

def _ocr_worker(args: tuple) -> tuple[int, str]:
    """Worker chạy Tesseract cho 1 trang (được gọi bởi ProcessPoolExecutor)."""
    idx, samples, width, height = args
    img = Image.frombytes("L", [width, height], samples)
    text = pytesseract.image_to_string(img, lang="vie", config="--oem 1 --psm 3").strip()
    return idx, text


# ─── API chính ────────────────────────────────────────────────────────────────

def batch_extract(pdf_path: str, cache_path: str = None, force: bool = False) -> list[dict]:
    """
    Trích xuất đặc trưng cho toàn bộ trang PDF.

    Trả về list[dict], mỗi phần tử ứng với 1 trang (theo thứ tự).
    Kết quả được cache tự động nếu cache_path được truyền vào.
    """
    if cache_path and not force:
        cached = _load_cache(cache_path, pdf_path)
        if cached is not None:
            print(f"✅ Dùng cache ({len(cached)} trang)")
            return cached

    t0 = time.time()

    # Pass 1: lấy text layer sẵn có
    with fitz.open(pdf_path) as doc:
        raw = [(i, doc[i].get_text("text").strip()) for i in range(len(doc))]

    has_text = {i: t for i, t in raw if len(re.sub(r'\s+', '', t)) >= TEXT_MIN_CHARS}
    scan_idx = [i for i, _ in raw if i not in has_text]
    print(f"  {len(has_text)} trang text-layer | {len(scan_idx)} trang scan")

    # Pass 2: render pixmap trong main process, sau đó OCR song song
    ocr_results: dict[int, str] = {}
    if scan_idx:
        pixmap_data: dict[int, tuple] = {}
        with fitz.open(pdf_path) as doc:
            for i in tqdm(scan_idx, desc="  Render trang scan", unit="trang"):
                pix = doc[i].get_pixmap(dpi=OCR_DPI, colorspace=fitz.csGRAY)
                pixmap_data[i] = (bytes(pix.samples), pix.width, pix.height)

        args = [(i, *pixmap_data[i]) for i in scan_idx]
        with concurrent.futures.ProcessPoolExecutor(max_workers=OCR_WORKERS) as executor:
            for idx, text in tqdm(
                executor.map(_ocr_worker, args),
                total=len(args), desc="  OCR", unit="trang"
            ):
                ocr_results[idx] = text

    features = [
        _build_features(i, has_text.get(i, ocr_results.get(i, "")))
        for i in range(len(raw))
    ]
    print(f"  ✅ Hoàn thành trong {time.time() - t0:.1f}s")

    if cache_path:
        _save_cache(cache_path, pdf_path, features)

    return features

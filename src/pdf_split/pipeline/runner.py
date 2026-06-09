"""
Pipeline chính của Module 4.1 — PDF Logical Splitting.

Luồng xử lý:
    [1] Kiểm tra file PDF
    [2] Trích xuất văn bản (text layer + OCR)
    [3] Phát hiện ranh giới → phân đoạn
    [4] Phân loại từng segment (rule → Qwen fallback)
    [5] Tách PDF theo segment → lưu ra thư mục output
    [6] Trả về payload JSON tổng hợp
"""
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import fitz

from config.settings import LABEL_SET, GT_SEGMENT_TYPES
from core import (
    batch_extract, get_segments,
    classify_rule, qwen_classify, qwen_is_available,
    extract_metadata,
)


# ─── Kiểm tra & phân loại từng segment ───────────────────────────────────────

def _classify_segment(
    pages: list[dict],
    use_qwen: bool,
    qwen_on: bool,
    gt_types: dict[int, str],
) -> dict:
    """Phân loại 1 segment và phát hiện nhãn nghi sai."""
    rule_label, rule_conf = classify_rule(pages)

    # Qwen verify khi confidence thấp
    qwen_label = None
    if use_qwen and qwen_on:
        combined_text = " ".join(p["text_full"] for p in pages[:2])[:1200]
        qwen_label, _ = qwen_classify(combined_text)

    if qwen_label and rule_conf < 0.80:
        label, conf = qwen_label, 0.88
    else:
        label, conf = rule_label, rule_conf

    # Phát hiện nhãn nghi sai
    gt_label = (gt_types or {}).get(pages[0]["page_num"])
    reasons  = []
    if qwen_label and qwen_label in LABEL_SET and label != qwen_label:
        severity = "HIGH" if conf >= 0.95 else "MED"
        reasons.append(f"[{severity}] rule≠qwen ({label}≠{qwen_label})")
    if conf < 0.70:
        reasons.append(f"low_conf={conf:.2f}")
    if gt_label and gt_label != label:
        reasons.append(f"gt={gt_label}≠pred={label}")

    return {
        "label"     : label,
        "confidence": round(conf, 3),
        "qwen_label": qwen_label,
        "reasons"   : reasons,
    }


# ─── Split PDF ────────────────────────────────────────────────────────────────

def _split_pdf(src_doc: fitz.Document, results: list[dict], run_dir: Path) -> list[dict]:
    """Tách PDF gốc thành các file con theo từng segment."""
    sub_docs = []
    for r in results:
        label_slug = r["type"].replace(" ", "_")
        fname = f"sub_{r['segment']:03d}_{label_slug}_p{r['page_start']}-{r['page_end']}.pdf"
        out   = fitz.open()
        out.insert_pdf(src_doc, from_page=r["page_start"] - 1, to_page=r["page_end"] - 1)
        out.save(str(run_dir / fname))
        out.close()
        sub_docs.append({
            "type"      : r["type"],
            "page_start": r["page_start"],
            "page_end"  : r["page_end"],
        })
    return sub_docs


# ─── API chính ────────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_path     : str,
    output_dir   : str,
    cache_path   : str  = None,
    gt_types     : dict = None,
    use_qwen     : bool = True,
    force_extract: bool = False,
) -> dict:
    """
    Chạy toàn bộ pipeline Module 4.1.

    Tham số:
        pdf_path      : đường dẫn đến file PDF đầu vào
        output_dir    : thư mục gốc lưu kết quả (mỗi lần chạy tạo thư mục con theo timestamp)
        cache_path    : đường dẫn file cache pickle (None = không dùng cache)
        gt_types      : dict ground truth {page_start: label} để phát hiện nhãn sai
        use_qwen      : có dùng Qwen verifier không
        force_extract : True = bỏ qua cache, extract lại từ đầu

    Trả về dict payload JSON, hoặc {"error_code": ..., "error_message": ...} nếu lỗi.
    """
    # Kiểm tra file đầu vào
    if not Path(pdf_path).exists():
        return {"error_code": "ERR_FILE_NOT_FOUND", "error_message": f"Không tìm thấy: {pdf_path}"}
    try:
        with fitz.open(pdf_path) as doc:
            if doc.is_encrypted:
                return {"error_code": "ERR_PDF_ENCRYPTED", "error_message": "PDF có mật khẩu."}
    except Exception as e:
        return {"error_code": "ERR_FILE_CORRUPTED", "error_message": str(e)}

    t0      = time.time()
    run_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    qwen_on = qwen_is_available() if use_qwen else False

    # ── Bước 1: Trích xuất ──────────────────────────────────────────────────
    print("📑 [1/4] Trích xuất văn bản...")
    features = batch_extract(pdf_path, cache_path=cache_path, force=force_extract)

    # ── Bước 2: Phân đoạn ───────────────────────────────────────────────────
    print("🔍 [2/4] Phát hiện ranh giới...")
    segments = get_segments(features)
    print(f"   → {len(segments)} segment")

    # ── Bước 3: Phân loại ───────────────────────────────────────────────────
    print("🏷️  [3/4] Phân loại...")
    results = []
    for i, pages in enumerate(segments):
        cls = _classify_segment(pages, use_qwen, qwen_on, gt_types or {})
        results.append({
            "segment"   : i + 1,
            "page_start": pages[0]["page_num"],
            "page_end"  : pages[-1]["page_num"],
            "type"      : cls["label"],
            "confidence": cls["confidence"],
            "qwen_type" : cls["qwen_label"],
            "mislabel"  : cls["reasons"],
            "metadata"  : extract_metadata(pages, cls["label"], cls["confidence"]),
        })

    # Báo cáo nhãn nghi sai
    flagged = [r for r in results if r["mislabel"]]
    if flagged:
        print(f"  🚨 {len(flagged)} segment nghi nhãn sai:")
        for r in flagged:
            icon = "🔴" if any("HIGH" in reason for reason in r["mislabel"]) else "🟡"
            print(f"    {icon} Seg {r['segment']} p{r['page_start']}-{r['page_end']}: {r['mislabel']}")
    else:
        print("  ✅ Không phát hiện nhãn sai")

    # ── Bước 4: Tách PDF ────────────────────────────────────────────────────
    print("✂️  [4/4] Tách PDF...")
    with fitz.open(pdf_path) as src_doc:
        sub_docs = _split_pdf(src_doc, results, run_dir)

    # ── Tổng hợp kết quả ────────────────────────────────────────────────────
    elapsed    = int((time.time() - t0) * 1000)
    avg_conf   = round(sum(r["confidence"] for r in results) / max(len(results), 1), 3)
    doc_types  = list({r["type"] for r in results})
    doc_id     = str(uuid.uuid4())[:8]

    payload = {
        "document_id"       : doc_id,
        "classification"    : doc_types[0] if len(doc_types) == 1 else "Tài liệu hỗn hợp",
        "confidence_overall": avg_conf,
        "processing_ms"     : elapsed,
        "mislabel_flags"    : len(flagged),
        "metadata"          : results[0]["metadata"] if results else [],
        "sub_documents"     : sub_docs,
    }

    with open(run_dir / f"result_{doc_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {len(sub_docs)} segment | conf={avg_conf} | {elapsed}ms")
    print(f"   Output: {run_dir}")
    return payload

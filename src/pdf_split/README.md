# Module 4.1 — PDF Logical Splitting

Tự động phát hiện ranh giới và phân loại văn bản pháp lý Việt Nam trong PDF hỗn hợp.

## Cấu trúc

```
module4_1/
├── main.py                  ← entrypoint, chạy từ đây
├── requirements.txt
├── config/
│   └── settings.py          ← toàn bộ cấu hình (đường dẫn, threshold, ground truth...)
├── core/
│   ├── extractor.py         ← trích xuất text (text layer + OCR song song)
│   ├── boundary.py          ← phát hiện ranh giới giữa các văn bản
│   ├── classifier.py        ← phân loại 4 nhãn (rule + Qwen fallback)
│   └── metadata.py          ← trích xuất số văn bản, ngày ký, cơ quan
└── pipeline/
    ├── runner.py             ← orchestrator, ghép 4 bước thành pipeline
    └── evaluator.py         ← so sánh kết quả với ground truth
```

## Cài đặt

```bash
pip install -r requirements.txt

# Tesseract + ngôn ngữ tiếng Việt (Ubuntu/Debian)
sudo apt install tesseract-ocr tesseract-ocr-vie
```

## Chạy

```bash
# Chạy với cấu hình mặc định trong config/settings.py
python main.py

# Các tuỳ chọn
python main.py --force          # bỏ qua cache, extract lại
python main.py --no-qwen        # tắt Qwen verifier
python main.py --no-eval        # không chạy evaluate
python main.py --pdf path/to/file.pdf --out path/to/output
```

## Cấu hình

Chỉnh sửa `config/settings.py`:

| Biến | Mô tả |
|------|-------|
| `PDF_PATH` | Đường dẫn file PDF đầu vào |
| `OUTPUT_DIR` | Thư mục lưu kết quả |
| `CACHE_PATH` | File cache pickle |
| `OCR_DPI` | DPI render trang scan (cao hơn = chậm hơn nhưng chính xác hơn) |
| `OCR_WORKERS` | Số process song song cho Tesseract |
| `QWEN_URL` / `QWEN_MODEL` | Endpoint và model Qwen (Ollama) |
| `GT_SEGMENT_TYPES` | Ground truth để phát hiện nhãn sai |

## Nhãn phân loại

- `Quyết định`
- `Bản án dân sự`
- `Bản án hình sự`
- `Bản án hành chính`

## Output

Mỗi lần chạy tạo thư mục `output/YYYYMMDD_HHMMSS/` chứa:
- `sub_001_Quyết_định_p1-8.pdf`, `sub_002_...` — các file PDF đã tách
- `result_<id>.json` — payload tổng hợp với metadata

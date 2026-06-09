"""
Entrypoint — chạy Module 4.1 từ command line:

    python main.py
    python main.py --force          # bỏ qua cache, extract lại
    python main.py --no-qwen        # tắt Qwen verifier
    python main.py --no-eval        # không chạy evaluate
    python main.py --pdf path/to/file.pdf --out path/to/output
"""
import argparse
import json

from config.settings import PDF_PATH, OUTPUT_DIR, CACHE_PATH, GT_SEGMENT_TYPES
from pipeline import run_pipeline, evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Module 4.1 — PDF Logical Splitting")
    p.add_argument("--pdf",      default=str(PDF_PATH),    help="Đường dẫn file PDF")
    p.add_argument("--out",      default=str(OUTPUT_DIR),  help="Thư mục output")
    p.add_argument("--cache",    default=str(CACHE_PATH),  help="Đường dẫn file cache")
    p.add_argument("--force",    action="store_true",       help="Bỏ qua cache, extract lại")
    p.add_argument("--no-qwen",  action="store_true",       help="Tắt Qwen verifier")
    p.add_argument("--no-eval",  action="store_true",       help="Không chạy evaluate")
    return p.parse_args()


def main():
    args = parse_args()

    result = run_pipeline(
        pdf_path      = args.pdf,
        output_dir    = args.out,
        cache_path    = args.cache,
        gt_types      = GT_SEGMENT_TYPES,
        use_qwen      = not args.no_qwen,
        force_extract = args.force,
    )

    print("\n" + "=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not args.no_eval and "error_code" not in result:
        print("\n" + "=" * 60)
        evaluate(result, GT_SEGMENT_TYPES)


if __name__ == "__main__":
    main()

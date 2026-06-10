"""
Demo Module 4.2 — Pipeline thật từ ảnh → Tesseract → Module 3 → qwen2.5:3b → Trích yếu

Chạy:
    python demo.py --images dataset/dataset/1.png dataset/dataset/27.png dataset/dataset/18.png
    python demo.py --images dataset/dataset/1.png dataset/dataset/27.png --workers 2
    python demo.py --folder dataset/dataset --workers 2 --output ketqua.json
    python demo.py --images dataset/dataset/1.png --length chi_tiet --debug
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from ocr.tesseract       import TesseractOCR
from ocr.adapter         import OCRAdapter
from ocr.confidence      import ConfidenceScorer
from src.utils.extractor import DocumentExtractor
from summarizer          import AutoSummarizer, SummaryLength, DocType

DEFAULT_IMAGES = [
    "dataset/dataset/27.png",
    "dataset/dataset/1.png",
    "dataset/dataset/18.png",
    "dataset/dataset/18.png"
]

W = 60

METADATA_LABELS = {
    'ten_loai_van_ban': 'Loai van ban',
    'so_van_ban':       'So van ban',
    'ky_hieu':          'Ky hieu',
    'ngay_thang_nam':   'Ngay ban hanh',
    'co_quan_ban_hanh': 'Co quan ban hanh',
}


def _sep(char: str = '-', width: int = W) -> str:
    return char * width


def _wrap(text: str, width: int = W - 4) -> list[str]:
    words, line, lines = text.split(), '', []
    for w in words:
        candidate = f'{line} {w}'.strip()
        if len(candidate) > width:
            if line:
                lines.append(line)
            line = w
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def process_image(
    image_path: str,
    ocr:        TesseractOCR,
    adapter:    OCRAdapter,
    scorer:     ConfidenceScorer,
    extractor:  DocumentExtractor,
    summ:       AutoSummarizer,
    length:     SummaryLength,
    debug:      bool = False,
) -> dict:

    name  = os.path.basename(image_path)

    # Pipeline (chạy im lặng, không print từng bước)
    image = cv2.imread(image_path)
    if image is None:
        print()
        print(_sep('='))
        print(f'  {name}')
        print(_sep('-'))
        print(f'  ERROR: Khong doc duoc anh')
        print(_sep('='))
        return {'image': name, 'error': 'Khong doc duoc anh'}

    group_result   = ocr.recognize(image)
    ocr_result     = adapter.convert(group_result)
    scored         = scorer.score(ocr_result)
    ocr_result     = scored if scored is not None else ocr_result

    if not ocr_result.raw_text.strip():
        print()
        print(_sep('='))
        print(f'  {name}')
        print(_sep('-'))
        print(f'  ERROR: OCR khong trich xuat duoc text')
        print(_sep('='))
        return {'image': name, 'error': 'OCR khong trich xuat duoc text'}

    module3_output = extractor.extract(ocr_result.raw_text, ocr_result.words)

    result = summ.summarize_from_ocr(
        ocr_result,
        length=length,
        document_id=name,
        module3_metadata=module3_output,
    )

    # In kết quả
    print()
    print(_sep('='))
    print(f'  {name}')
    print(_sep('-'))

    # # Metadata (Module 3)
    # if result.metadata:
    #     for m in result.metadata:
    #         label = METADATA_LABELS.get(m.field_name, m.field_name)
    #         print(f'  {label:<22} {m.value}')

    # print(_sep('-'))

    # Trích yếu (Module 4.2)
    print('  Trich yeu:')
    for ln in _wrap(result.summary):
        print(f'  {ln}')

    if debug:
        print(_sep('-'))
        print(f'  {"Do tin cay":<22} {result.confidence_overall:.0%}')
        print(f'  {"Thoi gian":<22} {result.processing_time_seconds:.2f}s')
        print(f'  {"OCR confidence":<22} {result.ocr_confidence:.2%}')
        print(f'  {"Flagged words":<22} {result.flagged_word_count}')

    print(_sep('='))
    return result.to_api_dict()


def process_batch_parallel(
    image_paths, ocr, adapter, scorer, extractor, summ,
    length, workers=2, debug=False,
) -> list[dict]:
    results = [None] * len(image_paths)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_image, path, ocr, adapter, scorer, extractor, summ, length, debug): i
            for i, path in enumerate(image_paths)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {'image': os.path.basename(image_paths[idx]), 'error': str(e)}
    return results


def main():
    parser = argparse.ArgumentParser(
        prog='demo',
        description='Module 4.2 — Anh -> Tesseract -> Module 3 -> LLM -> Trich yeu',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python demo.py --images dataset/dataset/1.png dataset/dataset/27.png dataset/dataset/18.png
  python demo.py --images dataset/dataset/1.png dataset/dataset/27.png --workers 2
  python demo.py --folder dataset/dataset --workers 2 --output ketqua.json
  python demo.py --images dataset/dataset/1.png --length chi_tiet --debug
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--images', '-i', nargs='+', metavar='FILE')
    group.add_argument('--folder', '-f', metavar='DIR')

    parser.add_argument('--length',  '-l', choices=['ngan', 'trung_binh', 'chi_tiet'], default='ngan')
    parser.add_argument('--model',   '-m', default='qwen2.5:3b')
    parser.add_argument('--workers', '-w', type=int, default=1)
    parser.add_argument('--output',  '-o', metavar='FILE', default=None)
    parser.add_argument('--debug',   '-d', action='store_true')
    args = parser.parse_args()

    if args.folder:
        folder = Path(args.folder)
        image_paths = sorted(
            str(p) for p in folder.iterdir()
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
        )
        if not image_paths:
            print(f'ERROR: Khong tim thay anh trong: {args.folder}')
            return
    elif args.images:
        image_paths = args.images
    else:
        image_paths = DEFAULT_IMAGES

    length = SummaryLength(args.length)

    ocr       = TesseractOCR()
    adapter   = OCRAdapter()
    scorer    = ConfidenceScorer(threshold=0.8)
    extractor = DocumentExtractor()
    summ      = AutoSummarizer(model=args.model)

    t_start = time.perf_counter()

    if args.workers > 1:
        all_results = process_batch_parallel(
            image_paths, ocr, adapter, scorer,
            extractor, summ, length, args.workers, args.debug,
        )
    else:
        all_results = [
            process_image(p, ocr, adapter, scorer, extractor, summ, length, args.debug)
            for p in image_paths
        ]

    total_time = time.perf_counter() - t_start
    success    = sum(1 for r in all_results if not r.get('error'))
    failed     = len(all_results) - success

    print()
    print(_sep('='))
    print(f'  {"Tong so anh":<20} {len(all_results)}')
    print(f'  {"Thanh cong":<20} {success}')
    print(f'  {"That bai":<20} {failed}')
    print(f'  {"Tong thoi gian":<20} {total_time:.1f}s')
    print(f'  {"Trung binh":<20} {total_time / len(all_results):.1f}s / anh')
    if failed:
        print(_sep('-'))
        for r in all_results:
            if r.get('error'):
                print(f'  {r.get("image", "?"):<30} {r["error"]}')
    print(_sep('='))

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f'\n  Saved -> {args.output}')

    print()


if __name__ == '__main__':
    main()